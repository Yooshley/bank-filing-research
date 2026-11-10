"""Local filing research demo. Run: python -m streamlit run app.py"""
from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv

from src.conversation_scope import ConversationState, resolve
from src.demo_backend import DemoBackend, safe_source_url
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
if 'conversation' not in st.session_state:
    st.session_state.conversation = ConversationState()
    st.session_state.messages = []
with st.sidebar:
    st.header('Research settings')
    if not os.getenv('OPENAI_API_KEY'):
        st.info('Add OPENAI_API_KEY to the project .env for generated answers. Local evidence search does not need it.')
    mode = st.selectbox('Workflow', ['Structured MMR', 'Basic top-k', 'Agent'])
    evidence_only = st.checkbox('Evidence only — no generation API calls', value=True)
    st.caption('Evidence-only search uses your configured embedding provider. OpenAI embeddings, if selected, still incur API calls.')
    if st.button('New conversation'):
        st.session_state.conversation = ConversationState()
        st.session_state.messages = []
        st.rerun()
    state = st.session_state.conversation
    st.write('Current banks:', ', '.join(state.banks) or 'None')
    st.write('Fiscal years:', ', '.join(map(str, state.years)) or 'None')
    st.caption('Supported: RBC, TD, BMO, BNS / Scotiabank, CIBC. Fiscal 2024–2025.')
    st.markdown('Try these in order:')
    st.code('What did BMO disclose about credit risk in 2025?\nWhat was its provision for credit losses?\nCompare TD with it.\nNow describe TD liquidity risk.\nWhat did Bank of Atlantis report in 2025?', language=None)


def display(message):
    with st.chat_message(message['role']):
        st.write(message['text'])
        if 'scope' in message:
            scope = message['scope']
            st.caption(f"Resolved scope: {', '.join(scope['banks'])} · {', '.join(map(str, scope['years']))}")
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
        if message.get('trace'):
            with st.expander('Agent tool actions and observations'):
                st.json(message['trace'])


for message in st.session_state.messages:
    display(message)
question = st.chat_input('Ask about a bank and fiscal year, or follow up on your last question', max_chars=2000)
if question:
    user = {'role': 'user', 'text': question}
    st.session_state.messages.append(user)
    display(user)
    scope = resolve(question, st.session_state.conversation)
    if not scope['ready']:
        reply = {'role': 'assistant', 'text': scope['message']}
    else:
        try:
            with st.spinner('Finding filing evidence…'):
                prepare_index()
                if 'backend' not in st.session_state:
                    st.session_state.backend = DemoBackend(BASE)
                backend = st.session_state.backend
                if evidence_only:
                    hits = backend.runtime.retrieve_scopes(scope['retrieval_query'], scope['scopes'], 'mmr', total=8)
                    result = {'answer': 'Retrieved evidence is shown below. Turn off evidence-only mode to generate a cited answer.', 'sources': hits}
                else:
                    result = backend.answer(scope, mode)
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
            reply = {'role': 'assistant', 'text': message + ' Conversation scope was not advanced.'}
    st.session_state.messages.append(reply)
    display(reply)
