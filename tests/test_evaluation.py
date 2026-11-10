"""Offline tests for the step 6 evaluation protocol and notebook repairs."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.evaluation import generation_stats, summarize, validate_judgment
from src.evaluation_dataset import make_dataset
from src.evaluation_runtime import Runtime, citation_check, token_cost
from src.notebook_utils import load_definitions

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_question_coverage_and_periods(self):
        questions = make_dataset(2025, 2024)
        self.assertEqual(len(questions), 30)
        self.assertEqual(len({q['id'] for q in questions}), 30)
        self.assertEqual({b for q in questions for b in q['banks']}, {'RBC','TD','BMO','BNS','CIBC'})
        self.assertEqual({kind:sum(q['question_type']==kind for q in questions) for kind in ['factual','comparative','trend','edge']},
                         {'factual':15,'comparative':5,'trend':5,'edge':5})
        for question in questions:
            if question['question_type']=='trend':
                self.assertEqual({s['fiscal_year'] for s in question['scopes']}, {2024,2025})

    def test_judge_rejects_invalid_scores(self):
        valid = {key:5 for key in ['faithfulness','relevance','completeness']}
        valid.update({key+'_reason':'Supported by the cited text.' for key in valid.copy()})
        valid['unsupported_claims']=[]
        self.assertEqual(validate_judgment(valid)['faithfulness'],5)
        for bad in [0,6,True,3.5,'5']:
            with self.assertRaises(ValueError):
                validate_judgment({**valid,'faithfulness':bad})

    def test_cost_cached_tokens_and_all_agent_calls(self):
        self.assertAlmostEqual(token_cost(1000,100,200), (.15*800+.075*200+.6*100)/1e6)
        events=[{'operation':'embedding','cost_usd':.001,'input_tokens':30,'seconds':.1},
                {'status':'success','cost_usd':.01,'input_tokens':100,'output_tokens':20},
                {'status':'success','cost_usd':.02,'input_tokens':200,'output_tokens':40}]
        measured=generation_stats(events)
        self.assertAlmostEqual(measured['generation_cost_usd'],.03)
        self.assertEqual(measured['input_tokens'],300)
        self.assertEqual(measured['chat_calls'],2)

    def test_failed_and_unjudged_cases_cannot_pass_thresholds(self):
        questions=make_dataset()
        cases=[]
        for question in questions:
            case={'config':'basic_top_k','question':question,'status':'complete',
                  'generation':{'tool_calls':0},'citation_checks':{'citation_identifiers_valid':True,'citation_present':True},
                  'embedding_cost_usd':0.,'generation_cost_usd':.001,'judge_cost_usd':.0005,
                  'latency_seconds':1.,'judgment':{'faithfulness':5,'relevance':5,'completeness':5}}
            cases.append(case)
        _,summary=summarize(cases)
        row=summary[summary['config']=='basic_top_k'].iloc[0]
        self.assertTrue(row['complete']);self.assertTrue(row['meets_quality_target'])
        self.assertAlmostEqual(row['annual_cost_1000_per_day_usd'],365.)
        cases[-1]['status']='error';cases[-1].pop('judgment')
        _,summary=summarize(cases)
        row=summary[summary['config']=='basic_top_k'].iloc[0]
        self.assertFalse(row['complete']);self.assertFalse(row['meets_quality_target'])
        self.assertEqual(row['scored_questions'],29);self.assertEqual(row['error_questions'],1)

    def test_citation_identifier_checks(self):
        self.assertTrue(citation_check('Supported [S1].',[{'chunk_id':'RBC_2025_annual_report_FY_1'}])['citation_identifiers_valid'])
        self.assertFalse(citation_check('Unsupported [S9].',[{'chunk_id':'real'}])['citation_identifiers_valid'])
        self.assertFalse(citation_check('RBC_2024_annual_report_FY_99999',[])['citation_identifiers_valid'])

    def test_loading_definitions_does_not_execute_examples(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'test.ipynb'
            path.write_text(json.dumps({'cells':[{'cell_type':'code','source':['def double(x): return 2*x\nraise RuntimeError("must not execute")']}]}))
            ns={};load_definitions(path,ns)
            self.assertEqual(ns['double'](3),6)

    def test_scotiabank_alias_and_schema_are_consistent(self):
        class Collection:
            def count(self):return 1
            def get(self, **kwargs):
                return {'metadatas':[{'bank_code':'BNS','fiscal_year':2025,'filing_id':'BNS_2025_FY',
                    'document_type':'annual_report','quarter':0,'file_name':'BNS_2025.pdf','source_url':'https://example.test'}]}
        ns={'np':np,'json':json,'TOP_K':4,'collection':Collection()}
        load_definitions(ROOT/'notebooks'/'05_agentic_workflow.ipynb', ns,
                         assignments=['TOOLS','FUNCTIONS','AGENT_INSTRUCTIONS'])
        self.assertEqual(ns['get_filing_list']('Scotiabank')['count'],1)
        for tool in ns['TOOLS']:
            enum=tool['function']['parameters']['properties'].get('bank',{}).get('enum',[])
            self.assertIn('BNS',enum)
            self.assertNotIn('Scotiabank',enum)
            properties=tool['function']['parameters']['properties']
            if 'document_type' in properties:
                self.assertEqual(properties['document_type']['enum'],['annual_report','AIF','quarterly'])

    def test_structured_metadata_is_resolved_from_source_not_generated(self):
        runtime=Runtime.__new__(Runtime)
        hit={'chunk_id':'BMO_2025_FY_1','bank':'Bank of Montreal','document_type':'annual_report',
             'fiscal_year':2025,'section':'Operational Risk','text':'The bank manages operational risk.'}
        runtime.retrieve_scopes=lambda *args,**kwargs:[hit]
        runtime.prompts={'step_by_step':'Use evidence.'};runtime.model='gpt-4o-mini'
        generated={'answer':'The bank manages operational risk [S1].','confidence':'high','caveats':[],
                   'sources':[{'label':'S1','bank':'BMO','document':'annual report','section':'risk','year':2024}]}
        runtime.chat=lambda **kwargs:SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(generated)))])
        result=runtime.rag({'question':'What risks?','scopes':[{'bank_code':'BMO'}]},structured=True)
        self.assertEqual(result['structured']['sources'][0]['bank'],'Bank of Montreal')
        self.assertEqual(result['structured']['sources'][0]['year'],2025)
        self.assertEqual(result['model_source_metadata'][0]['year'],2024)
        self.assertEqual(len(result['metadata_corrections']),4)


if __name__=='__main__':unittest.main()
