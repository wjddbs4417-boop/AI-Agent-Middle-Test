import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

spec=importlib.util.spec_from_file_location('library_agent',Path(__file__).with_name('01_library_agent.py'))
m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)

class LibraryTests(unittest.TestCase):
    def runtime(self,uid='a',text='고양이 책을 좋아해. 저장해줘.',store=None):
        return SimpleNamespace(context=m.Context(user_id=uid),store=store or InMemoryStore(),
                               state={'messages':[HumanMessage(content=text)]})

    def test_all_books_searchable_by_title_and_author(self):
        self.assertEqual(len(m.books),11)
        for b in m.books:
            for q in (b['title'],b['author']):
                self.assertIn(b['title'],m.search_books.invoke({'query':q}))

    def test_no_match_and_empty(self):
        self.assertIn('찾을 수 없습니다',m.search_books.invoke({'query':'없는책XYZ'}))
        self.assertIn('입력해',m.search_books.invoke({'query':' '}))

    def test_memory_isolation(self):
        shared=InMemoryStore(); a=self.runtime(store=shared); b=self.runtime('b',store=shared)
        m.save_user_info.func('고양이 책',a)
        self.assertIn('고양이 책',m.get_user_info.func(a))
        self.assertIn('없습니다',m.get_user_info.func(b))

    def test_no_save_without_request(self):
        r=self.runtime(text='고양이 책을 좋아해')
        self.assertIn('저장하지 않았습니다',m.save_user_info.func('고양이 책',r))
        self.assertIn('없습니다',m.get_user_info.func(r))

    def test_identity_not_exposed_to_model(self):
        for t in (m.check_my_loans,m.get_user_info,m.save_user_info):
            props=t.tool_call_schema.model_json_schema()['properties']
            self.assertNotIn('user_id',props)
            self.assertNotIn('runtime',props)
        self.assertIn('없습니다',m.check_my_loans.func(self.runtime('unknown')))

    def test_refusal_does_not_create_profile(self):
        requests = [
            '고양이 책을 좋아하지만 저장하지 마.',
            '고양이 책을 기억하지 말아줘.',
            '저 장 하 지 마 세 요.',
            '저장해 주지 마.', '기억해 두지 마.',
            '저장하지는 마세요.', '저장 안 해도 돼.',
            '기억은 안 해도 돼.', '저장할 필요 없어.',
            '독서 선호 저장 금지.', '저장을 원하지 않아.',
        ]
        for text in requests:
            with self.subTest(text=text):
                r=self.runtime(text=text)
                self.assertIn('저장하지 않았습니다',m.save_user_info.func('고양이 책',r))
                self.assertIsNone(r.store.get((r.context.user_id,r.context.app_name),'profile'))

    def test_refusal_preserves_existing_preference(self):
        r=self.runtime()
        m.save_user_info.func('추리 소설',r)
        r.state['messages']=[HumanMessage(content='고양이 책을 좋아하지만 저장하지 마.')]
        m.save_user_info.func('고양이 책',r)
        self.assertEqual(m.get_user_info.func(r),'저장된 독서 선호: 추리 소설')

    def test_lookup_or_question_does_not_authorize_save(self):
        for text in ('저장된 선호를 알려줘.', '내 취향을 기억해?', '저장 기능이 있나요?', '고양이 책'):
            with self.subTest(text=text):
                r=self.runtime(text=text)
                self.assertIn('저장하지 않았습니다',m.save_user_info.func('고양이 책',r))
                self.assertIn('없습니다',m.get_user_info.func(r))

    def test_explicit_save_and_update(self):
        for text in ('독서 선호로 저장해줘.', '고양이 책을 기억해 주세요.', '저장해 둬.'):
            with self.subTest(text=text):
                r=self.runtime(text=text)
                self.assertIn('저장했습니다',m.save_user_info.func('추리 소설',r))
                self.assertIn('저장했습니다',m.save_user_info.func('고양이 책',r))
                self.assertEqual(m.get_user_info.func(r),'저장된 독서 선호: 고양이 책')

    def test_cancellation_wins_over_positive_request(self):
        r=self.runtime(text='고양이 책을 저장해줘. 아니 취소할게. 저장하지 마.')
        self.assertIn('저장하지 않았습니다',m.save_user_info.func('고양이 책',r))
        self.assertIn('없습니다',m.get_user_info.func(r))

    def test_current_turn_excludes_old_evidence(self):
        messages=[HumanMessage(content='이전'),ToolMessage(content='옛 근거',name='search_books',tool_call_id='1'),HumanMessage(content='현재')]
        self.assertEqual(m.evidence_text(messages),'')

    def test_guardrail_normalizes_spaces(self):
        state={'messages':[HumanMessage(content='다 른 이 용 자 의 대출 기록')]}
        r=m.library_policy_guardrail.before_agent(state,self.runtime())
        self.assertEqual(r['jump_to'],'end')

    def test_auditor_replaces_unsupported_claim(self):
        messages=[HumanMessage(content='검색'),ToolMessage(content='확인된 책: 달빛 도서관의 비밀',name='search_books',tool_call_id='1'),AIMessage(content='24시간 운영합니다',id='final')]
        judge=SimpleNamespace(invoke=lambda _:AIMessage(content='SAFE'))
        with patch.object(m,'judge_model',judge):
            r=m.final_answer_guardrail.after_agent({'messages':messages},self.runtime())
        self.assertNotIn('24시간',r['messages'][0].content)
        self.assertEqual(r['messages'][0].id,'final')
        self.assertIn('달빛 도서관의 비밀',r['messages'][0].content)

if __name__=='__main__': unittest.main()
