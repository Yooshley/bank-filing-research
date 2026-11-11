"""Local filing research demo. Run: python -m streamlit run app.py"""
from pathlib import Path
import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.conversation_interpreter import ConversationState, resolve
from src.demo_backend import DemoBackend, safe_source_url, load_search_resources
from src.cloud_index import ensure_cloud_index

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env', encoding='utf-8-sig')
load_dotenv(BASE / 'notebooks' / '.env', encoding='utf-8-sig')
try:
    for name in ('OPENAI_API_KEY', 'HF_TOKEN', 'INDEX_BUNDLE_URL', 'INDEX_BUNDLE_SHA256'):
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
except FileNotFoundError:
    pass
st.set_page_config(page_title='Bank Filing Research', page_icon='📚', layout='wide')
st.title('Bank Filing Research')
st.caption('Explore Canadian bank disclosures with cited evidence. Verify financial claims against the original filing.')

@st.cache_resource(show_spinner='Preparing filing index…')
def prepare_index():
    ensure_cloud_index(BASE)
    return True

@st.cache_resource(show_spinner='Preparing search for this server…')
def prepare_search_resources(index_version):
    prepare_index()
    return load_search_resources(BASE)

def prepare_backend():
    if 'backend' not in st.session_state:
        resources = prepare_search_resources(os.getenv('INDEX_BUNDLE_SHA256', 'local'))
        st.session_state.backend = DemoBackend(BASE, search_resources=resources)
    return st.session_state.backend

if st.session_state.get('conversation_version') != 4:
    st.session_state.conversation = ConversationState()
    st.session_state.messages = []
    st.session_state.conversation_version = 4
    st.session_state.pop('backend', None)
with st.sidebar:
    st.header('Research settings')
    if not os.getenv('OPENAI_API_KEY'):
        st.info('Configure OPENAI_API_KEY to interpret conversations and generate answers.')
    mode = st.selectbox('Workflow', ['Research assistant', 'Basic top-k (baseline)'])
    st.caption('Research assistant searches by topic and keywords, then checks each answer point against its evidence.')
    evidence_only = st.checkbox('Evidence Only', value=False)
    st.caption('Evidence Only skips answer generation. Conversation interpretation still uses the API.')
    if st.button('Prepare demo'):
        try:
            prepare_backend()
            st.success('Search is ready. No answer-generation calls were made.')
        except Exception:
            st.error('Search preparation failed. Check the index bundle and local model configuration, then retry.')
    st.caption('Prepare demo before presenting to load search ahead of the first question.')
    if st.button('New conversation'):
        st.session_state.conversation = ConversationState()
        st.session_state.messages = []
        st.rerun()
    state = st.session_state.conversation
    st.write('Current banks:', ', '.join(state.banks) or 'None')
    st.write('Fiscal years:', ', '.join(map(str, state.years)) or 'None')
    st.caption('Supported: RBC, TD, BMO, BNS / Scotiabank, CIBC. Fiscal 2024–2025.')


def display(message):
    with st.chat_message(message['role']):
        st.write(message['text'])
        if 'scope' in message:
            scope = message['scope']
            st.caption(f"Resolved scope: {', '.join(scope['banks'])} · {', '.join(map(str, scope['years']))}")
            with st.expander('Understood research question'):
                st.text(scope.get('standalone_question', scope['question']))
        if message.get('cited_labels'):
            st.caption('Answer source references: ' + ', '.join(message['cited_labels']))
        for i, hit in enumerate(message.get('sources', []), 1):
            with st.expander(f"S{i} · {hit.get('bank_code')} · {hit.get('fiscal_year')} · {hit.get('section', 'Section unknown')}"):
                st.caption(f"{hit.get('document_type')} · pages {hit.get('page_start')}–{hit.get('page_end')} · {hit['chunk_id']}")
                st.text(hit['text'])
                url = safe_source_url(hit.get('source_url'))
                if url:
                    st.link_button('Open original filing', url)
        if 'seconds' in message:
            st.caption(f"{message['seconds']} seconds · estimated API cost US${message.get('api_cost_usd', 0):.6f}")
        if message.get('cache_hit'):
            st.caption('Reused a verified answer for this exact research request.')
        if message.get('trace'):
            with st.expander('Research steps'):
                st.json(message['trace'])


for message in st.session_state.messages:
    display(message)
question = st.chat_input('Ask about a bank and fiscal year, or follow up on your last question', max_chars=2000)
if question:
    started = time.perf_counter()
    user = {'role': 'user', 'text': question}
    st.session_state.messages.append(user)
    display(user)
    try:
        with st.spinner('Understanding your question…'):
            scope = resolve(question, st.session_state.conversation)
    except Exception as exc:
        scope = {'ready': False, 'message': 'I couldn’t confidently resolve that reference. Please name the bank and year, or rephrase the question. Your previous research request is still saved.'}
    if not scope['ready']:
        reply = {'role': 'assistant', 'text': scope['message']}
        if scope.get('interpretation_cost_usd'):
            reply.update(seconds=round(time.perf_counter()-started, 2), api_cost_usd=scope['interpretation_cost_usd'])
    else:
        try:
            with st.spinner('Finding filing evidence…'):
                backend = prepare_backend()
                if evidence_only:
                    result = backend.retrieve(scope)
                else:
                    result = backend.answer(scope, mode)
                result['api_cost_usd'] = result.get('api_cost_usd', 0) + scope.get('interpretation_cost_usd', 0)
                result['seconds'] = round(time.perf_counter()-started, 2)
            st.session_state.conversation.accept(scope)
            reply = {'role': 'assistant', 'text': result.pop('answer'), 'scope': scope, **result}
        except Exception as exc:
            # SDK exceptions can include credentials; expose only the exception type.
            if type(exc).__name__ == 'NotFoundError':
                message = 'The saved Chroma collection is missing. Run notebooks 1 and 2 in order to rebuild the local filing index, then retry.'
            elif not evidence_only and not os.getenv('OPENAI_API_KEY'):
                message = 'Add OPENAI_API_KEY to the project .env, then restart the app to generate answers.'
            else:
                message = f'Research could not complete ({type(exc).__name__}). Check the local model cache, index and API configuration.'
            reply = {'role': 'assistant', 'text': message + ' Your research request was saved; you can retry or clarify it.'}
    st.session_state.messages.append(reply)
    display(reply)
