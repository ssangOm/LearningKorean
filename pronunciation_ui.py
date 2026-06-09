from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.pronunciation.config import REFERENCE_MANIFEST_PATH, USER_TEST_DIR
from src.pronunciation.ai_coach import (
    AiCoachError,
    clean_model_text,
    generate_ai_coaching,
    generate_conversation_feedback,
    generate_conversation_prompt,
)
from src.pronunciation.asr import DEFAULT_ASR_MODEL_SIZE, transcribe_audio
from src.pronunciation.evaluate import EvaluationResult, evaluate_file, write_results_csv
from src.pronunciation.realtime_test import record_microphone_until_silence


RESULT_OUTPUT_PATH = Path("outputs/pronunciation/realtime_result.csv")
HUMAN_REFERENCE_MANIFEST_PATH = Path("outputs/pronunciation/human_reference_manifest.csv")
DEFAULT_UI_ASR_MODEL_SIZE = DEFAULT_ASR_MODEL_SIZE
PRIMARY_FONT_SIZE = 16
TITLE_FONT_SIZE = 32
SUBTITLE_FONT_SIZE = 17
PROMPT_FONT_SIZE = 28
CONVERSATION_PROMPT_FONT_SIZE = 26
METRIC_FONT_SIZE = 34
DETAIL_PROMPT_MIN_HEIGHT = 124
METRIC_CARD_MIN_HEIGHT = 112
RESULT_LINE_MIN_HEIGHT = 56
AI_COACHING_MIN_HEIGHT = 220
SENTENCE_LIST_FONT_SIZE = 14
SENTENCE_LIST_MIN_WIDTH = 600
SENTENCE_LIST_PAGE_INDEX = 0
SENTENCE_DETAIL_PAGE_INDEX = 1
INITIAL_CONVERSATION_PROMPT_TEXT = "대화 탭을 열면 질문을 자동으로 준비합니다."
CONVERSATION_NEW_PROMPT_BUTTON_TEXT = "새 질문"


def choose_reference_manifest(
    human_manifest_path: str | Path = HUMAN_REFERENCE_MANIFEST_PATH,
    fallback_manifest_path: str | Path = REFERENCE_MANIFEST_PATH,
) -> Path:
    human_manifest_path = Path(human_manifest_path)
    fallback_manifest_path = Path(fallback_manifest_path)
    if human_manifest_path.exists() and human_manifest_path.stat().st_size > 0:
        return human_manifest_path
    return fallback_manifest_path


def load_sentence_options(
    manifest_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    manifest_path = choose_reference_manifest() if manifest_path is None else Path(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required_columns = {"sentence_id", "text"}
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        joined = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest is missing required columns: {joined}")

    unique = manifest[["sentence_id", "text"]].drop_duplicates("sentence_id")
    unique = unique.sort_values("sentence_id")
    return [(str(row.sentence_id), str(row.text)) for row in unique.itertuples(index=False)]


def build_recording_path(sentence_id: str, output_dir: str | Path = USER_TEST_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"ui_{sentence_id}.wav"


def build_conversation_recording_path(output_dir: str | Path = USER_TEST_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "ui_conversation_response.wav"


def format_sentence_option_label(index: int) -> str:
    return f"문장 {index + 1:02d}"


def format_sentence_list_item_label(index: int, sentence_id: str, text: str) -> str:
    return f"{index + 1:02d}  {text}"


def format_result_summary(result: EvaluationResult, ai_coaching: str | None = None) -> str:
    recognized_line = (
        f"인식된 문장: {result.recognized_text}\n"
        if result.recognized_text
        else ""
    )
    return (
        f"점수: {result.score:.1f} / 100\n"
        f"텍스트 일치도: {result.text_score:.1f} / 100\n"
        f"{recognized_line}"
        f"정답 문장: {result.closest_sentence_id} - {result.closest_text}"
    )


def format_ai_coaching(coaching: str | None) -> str:
    cleaned = clean_model_text(coaching or "")
    return cleaned if cleaned else "AI 코칭 결과가 여기에 표시됩니다."


def format_conversation_feedback(feedback: str | None) -> str:
    cleaned = clean_model_text(feedback or "")
    return cleaned if cleaned else "대화 평가가 여기에 표시됩니다."


def should_auto_start_conversation_prompt(current_prompt: str | None, prompt_worker: object | None) -> bool:
    return not current_prompt and prompt_worker is None


def is_new_conversation_prompt(prompt: str, previous_prompts: list[str] | tuple[str, ...]) -> bool:
    normalized_prompt = _normalize_prompt_for_comparison(prompt)
    previous = {_normalize_prompt_for_comparison(item) for item in previous_prompts}
    return normalized_prompt not in previous


def _normalize_prompt_for_comparison(prompt: str) -> str:
    return " ".join(clean_model_text(prompt).split())


def build_stylesheet() -> str:
    return f"""
    QWidget {{
        background: #f6f8fb;
        color: #18212f;
        font-family: Helvetica, Arial, sans-serif;
        font-size: {PRIMARY_FONT_SIZE}px;
    }}
    QLabel#title {{
        font-size: {TITLE_FONT_SIZE}px;
        font-weight: 700;
    }}
    QLabel#subtitle {{
        color: #52616f;
        font-size: {SUBTITLE_FONT_SIZE}px;
    }}
    QLabel#prompt, QLabel#conversationPrompt {{
        background: #ffffff;
        border: 1px solid #d8dee6;
        border-radius: 8px;
        font-size: {PROMPT_FONT_SIZE}px;
        font-weight: 700;
        min-height: {DETAIL_PROMPT_MIN_HEIGHT}px;
        padding: 18px;
    }}
    QLabel#conversationPrompt {{
        font-size: {CONVERSATION_PROMPT_FONT_SIZE}px;
    }}
    QLabel#metricValue {{
        font-size: {METRIC_FONT_SIZE}px;
        font-weight: 700;
    }}
    QLabel#metricTitle, QLabel#status {{
        color: #52616f;
    }}
    QTextEdit#aiCoaching {{
        background: #ffffff;
        color: #1f2933;
        border: 1px solid #d8dee6;
        border-radius: 8px;
        font-size: {PRIMARY_FONT_SIZE}px;
        padding: 14px;
    }}
    QLabel#resultLine {{
        color: #1f2933;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        min-height: {RESULT_LINE_MIN_HEIGHT}px;
        padding: 10px;
    }}
    QFrame#metricCard {{
        background: #ffffff;
        border: 1px solid #d8dee6;
        border-radius: 8px;
        min-height: {METRIC_CARD_MIN_HEIGHT}px;
        padding: 14px;
    }}
    QScrollArea#detailScroll {{
        border: 0;
        background: transparent;
    }}
    QFrame#practicePanel {{
        background: transparent;
    }}
    QListWidget#sentenceList {{
        background: #ffffff;
        border: 1px solid #d8dee6;
        border-radius: 8px;
        font-size: {SENTENCE_LIST_FONT_SIZE}px;
        padding: 8px;
        outline: 0;
    }}
    QListWidget#sentenceList::item {{
        border-radius: 6px;
        padding: 12px;
        margin: 3px;
    }}
    QListWidget#sentenceList::item:selected {{
        background: #dbeafe;
        color: #12326b;
    }}
    QTabWidget::pane {{
        border: 0;
    }}
    QTabBar::tab {{
        background: #e9eef5;
        border-radius: 6px;
        color: #334155;
        padding: 11px 18px;
        margin-right: 6px;
    }}
    QTabBar::tab:selected {{
        background: #1f5fbf;
        color: #ffffff;
    }}
    QTextEdit {{
        background: #ffffff;
        border: 1px solid #cbd5df;
        border-radius: 6px;
        padding: 10px;
    }}
    QPushButton {{
        background: #1f5fbf;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-size: {PRIMARY_FONT_SIZE}px;
        padding: 12px 20px;
        font-weight: 700;
    }}
    QPushButton:disabled {{
        background: #9aa8b8;
    }}
    """


class EvaluationWorker(QThread):
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(
        self,
        sentence_id: str,
        output_path: Path,
        manifest_path: Path,
    ) -> None:
        super().__init__()
        self.sentence_id = sentence_id
        self.output_path = output_path
        self.manifest_path = manifest_path
        self.asr_model_size = DEFAULT_UI_ASR_MODEL_SIZE

    def run(self) -> None:
        try:
            record_microphone_until_silence(self.output_path)
            result = evaluate_file(
                self.output_path,
                sentence_id=self.sentence_id,
                manifest_path=self.manifest_path,
                use_asr=True,
                asr_model_size=self.asr_model_size,
            )
            write_results_csv([result], RESULT_OUTPUT_PATH)
        except Exception as exc:  # UI boundary: display instead of crashing.
            self.failed.emit(str(exc))
            return

        self.completed.emit(result, self.output_path)


class AiCoachWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, result: EvaluationResult) -> None:
        super().__init__()
        self.result = result

    def run(self) -> None:
        try:
            coaching = generate_ai_coaching(self.result)
        except AiCoachError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # UI boundary: keep AI failures non-fatal.
            self.failed.emit(f"AI 코칭 생성 중 오류가 발생했습니다: {exc}")
            return

        self.completed.emit(coaching)


class ConversationPromptWorker(QThread):
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, previous_prompts: list[str] | tuple[str, ...] | None = None) -> None:
        super().__init__()
        self.previous_prompts = list(previous_prompts or [])

    def run(self) -> None:
        try:
            prompt = ""
            for _attempt in range(3):
                prompt = generate_conversation_prompt(previous_prompts=self.previous_prompts)
                if is_new_conversation_prompt(prompt, self.previous_prompts):
                    break
                self.previous_prompts.append(prompt)
        except AiCoachError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # UI boundary: keep AI failures non-fatal.
            self.failed.emit(f"대화 질문 생성 중 오류가 발생했습니다: {exc}")
            return

        self.completed.emit(prompt)


class ConversationResponseWorker(QThread):
    completed = Signal(str, str)
    failed = Signal(str)

    def __init__(self, prompt_text: str, output_path: Path) -> None:
        super().__init__()
        self.prompt_text = prompt_text
        self.output_path = output_path

    def run(self) -> None:
        try:
            record_microphone_until_silence(self.output_path)
            transcription = transcribe_audio(self.output_path, model_size=DEFAULT_UI_ASR_MODEL_SIZE)
            feedback = generate_conversation_feedback(self.prompt_text, transcription.text)
        except AiCoachError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # UI boundary: display instead of crashing.
            self.failed.emit(f"대화 평가 중 오류가 발생했습니다: {exc}")
            return

        self.completed.emit(transcription.text, feedback)


class PronunciationWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Korean Pronunciation Coach")
        self.setMinimumSize(900, 680)

        self.manifest_path = choose_reference_manifest()
        self.sentence_options = load_sentence_options(self.manifest_path)
        if not self.sentence_options:
            raise RuntimeError("No sentence options found in reference manifest.")

        self.worker: EvaluationWorker | None = None
        self.ai_worker: AiCoachWorker | None = None
        self.conversation_prompt_worker: ConversationPromptWorker | None = None
        self.conversation_response_worker: ConversationResponseWorker | None = None
        self.latest_result: EvaluationResult | None = None
        self.latest_ai_coaching: str | None = None
        self.current_conversation_prompt: str | None = None
        self.conversation_prompt_history: list[str] = []
        self._build_ui()
        self._update_prompt()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(22, 22, 22, 22)
        root_layout.setSpacing(16)

        title = QLabel("Korean Pronunciation Coach")
        title.setObjectName("title")
        subtitle = QLabel("로컬 Whisper 음성 인식으로 정답 문장과 인식 문장의 일치도를 평가합니다.")
        subtitle.setObjectName("subtitle")

        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)
        self.manifest_label = QLabel(f"Reference: {self.manifest_path}")
        self.manifest_label.setObjectName("status")
        root_layout.addWidget(self.manifest_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_sentence_practice_tab(), "문장 연습")
        self.tabs.addTab(self._build_conversation_practice_tab(), "대화 연습")
        self.tabs.currentChanged.connect(self._handle_tab_changed)
        root_layout.addWidget(self.tabs)

        footer = QHBoxLayout()
        root_layout.addLayout(footer)

        self.status_label = QLabel("문장을 선택하고 녹음 및 평가를 누르세요.")
        self.status_label.setObjectName("status")
        footer.addWidget(self.status_label, stretch=1)

        self.setStyleSheet(build_stylesheet())

    def _build_sentence_practice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        self.sentence_stack = QStackedWidget()
        layout.addWidget(self.sentence_stack)

        self.sentence_stack.addWidget(self._build_sentence_list_page())
        self.sentence_stack.addWidget(self._build_sentence_detail_page())
        self.sentence_stack.setCurrentIndex(SENTENCE_LIST_PAGE_INDEX)

        return tab

    def _build_sentence_list_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("문장 선택")
        title.setObjectName("prompt")
        title.setText("연습할 문장을 선택하세요.")
        layout.addWidget(title)

        self.sentence_list = QListWidget()
        self.sentence_list.setObjectName("sentenceList")
        self.sentence_list.setMinimumWidth(SENTENCE_LIST_MIN_WIDTH)
        for index, (sentence_id, text) in enumerate(self.sentence_options):
            item = QListWidgetItem(format_sentence_list_item_label(index, sentence_id, text))
            item.setData(Qt.ItemDataRole.UserRole, sentence_id)
            self.sentence_list.addItem(item)
        self.sentence_list.itemClicked.connect(self._open_sentence_detail_page)
        self.sentence_list.itemActivated.connect(self._open_sentence_detail_page)
        layout.addWidget(self.sentence_list)

        self.sentence_list.setCurrentRow(0)
        return page

    def _build_sentence_detail_page(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setObjectName("detailScroll")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        header = QHBoxLayout()
        layout.addLayout(header)
        self.back_to_sentence_list_button = QPushButton("문장 목록")
        self.back_to_sentence_list_button.clicked.connect(self._show_sentence_list_page)
        header.addWidget(self.back_to_sentence_list_button)
        header.addStretch(1)
        self.ai_button = QPushButton("AI 코칭 생성")
        self.ai_button.setEnabled(False)
        self.ai_button.clicked.connect(self.start_ai_coaching)
        self.record_button = QPushButton("녹음 및 평가")
        self.record_button.clicked.connect(self.start_recording)
        header.addWidget(self.ai_button)
        header.addWidget(self.record_button)

        practice_panel = QFrame()
        practice_panel.setObjectName("practicePanel")
        practice_layout = QVBoxLayout(practice_panel)
        practice_layout.setSpacing(14)
        layout.addWidget(practice_panel)

        self.prompt_label = QLabel()
        self.prompt_label.setObjectName("prompt")
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setMinimumHeight(DETAIL_PROMPT_MIN_HEIGHT)
        self.prompt_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        prompt_title = QLabel("읽을 문장")
        prompt_title.setObjectName("metricTitle")
        practice_layout.addWidget(prompt_title)
        practice_layout.addWidget(self.prompt_label)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(14)
        practice_layout.addLayout(metrics_layout)

        self.score_value = QLabel("-")
        self.text_score_value = QLabel("-")
        metrics_layout.addWidget(self._metric_card("점수", self.score_value))
        metrics_layout.addWidget(self._metric_card("텍스트 일치도", self.text_score_value))

        self.ai_coaching_text = QTextEdit()
        self.ai_coaching_text.setObjectName("aiCoaching")
        self.ai_coaching_text.setReadOnly(True)
        self.ai_coaching_text.setMinimumHeight(AI_COACHING_MIN_HEIGHT)
        self.ai_coaching_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.ai_coaching_text.setText(format_ai_coaching(None))
        practice_layout.addWidget(self._ai_coaching_panel())

        self.recognized_label = QLabel("인식된 문장: -")
        self.recognized_label.setObjectName("resultLine")
        self.recognized_label.setWordWrap(True)
        self.recognized_label.setMinimumHeight(RESULT_LINE_MIN_HEIGHT)
        self.target_label = QLabel("정답 문장: -")
        self.target_label.setObjectName("resultLine")
        self.target_label.setWordWrap(True)
        self.target_label.setMinimumHeight(RESULT_LINE_MIN_HEIGHT)
        practice_layout.addWidget(self.recognized_label)
        practice_layout.addWidget(self.target_label)

        layout.addStretch(1)
        scroll_area.setWidget(page)
        return scroll_area

    def _build_conversation_practice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)

        self.conversation_prompt_label = QLabel(INITIAL_CONVERSATION_PROMPT_TEXT)
        self.conversation_prompt_label.setObjectName("conversationPrompt")
        self.conversation_prompt_label.setWordWrap(True)
        layout.addWidget(self.conversation_prompt_label)

        self.conversation_recognized_label = QLabel("내 답변: -")
        self.conversation_recognized_label.setObjectName("resultLine")
        self.conversation_recognized_label.setWordWrap(True)
        layout.addWidget(self.conversation_recognized_label)

        self.conversation_feedback_text = QTextEdit()
        self.conversation_feedback_text.setObjectName("aiCoaching")
        self.conversation_feedback_text.setReadOnly(True)
        self.conversation_feedback_text.setMinimumHeight(320)
        self.conversation_feedback_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.conversation_feedback_text.setText(format_conversation_feedback(None))
        layout.addWidget(self.conversation_feedback_text)

        conversation_actions = QHBoxLayout()
        layout.addLayout(conversation_actions)
        self.new_conversation_button = QPushButton(CONVERSATION_NEW_PROMPT_BUTTON_TEXT)
        self.new_conversation_button.clicked.connect(self.start_conversation_prompt)
        self.answer_conversation_button = QPushButton("답변 녹음 및 평가")
        self.answer_conversation_button.setEnabled(False)
        self.answer_conversation_button.clicked.connect(self.start_conversation_answer)
        conversation_actions.addStretch(1)
        conversation_actions.addWidget(self.new_conversation_button)
        conversation_actions.addWidget(self.answer_conversation_button)

        return tab

    def _metric_card(self, title: str, value_label: QLabel) -> QFrame:
        card = QFrame()
        card.setObjectName("metricCard")
        card.setMinimumHeight(METRIC_CARD_MIN_HEIGHT)
        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label.setObjectName("metricValue")
        value_label.setMinimumHeight(48)
        value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card

    def _ai_coaching_panel(self) -> QFrame:
        panel = QFrame()
        layout = QVBoxLayout(panel)
        title_label = QLabel("AI 코칭")
        title_label.setObjectName("metricTitle")
        layout.addWidget(title_label)
        layout.addWidget(self.ai_coaching_text)
        return panel

    def _handle_tab_changed(self, index: int) -> None:
        conversation_tab_index = 1
        if index != conversation_tab_index:
            return
        if not should_auto_start_conversation_prompt(self.current_conversation_prompt, self.conversation_prompt_worker):
            return
        self.start_conversation_prompt()

    def _selected_sentence_id(self) -> str:
        item = self.sentence_list.currentItem()
        if item is None:
            return self.sentence_options[0][0]
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _open_sentence_detail_page(self, _item: QListWidgetItem | None = None) -> None:
        self._update_prompt()
        self._reset_sentence_result_panel()
        self.sentence_stack.setCurrentIndex(SENTENCE_DETAIL_PAGE_INDEX)
        self.status_label.setText("녹음 및 평가를 누르세요.")

    def _show_sentence_list_page(self) -> None:
        self.sentence_stack.setCurrentIndex(SENTENCE_LIST_PAGE_INDEX)
        self.status_label.setText("연습할 문장을 선택하세요.")

    def _reset_sentence_result_panel(self) -> None:
        self.latest_result = None
        self.latest_ai_coaching = None
        self.score_value.setText("-")
        self.text_score_value.setText("-")
        self.ai_button.setEnabled(False)
        self.ai_coaching_text.setText(format_ai_coaching(None))
        self.recognized_label.setText("인식된 문장: -")
        self.target_label.setText(f"정답 문장: {self.prompt_label.text() or '-'}")

    def _update_prompt(self, _index: int | None = None) -> None:
        index = self.sentence_list.currentRow()
        if index < 0:
            return
        self.prompt_label.setText(self.sentence_options[index][1])

    def start_recording(self) -> None:
        sentence_id = self._selected_sentence_id()
        output_path = build_recording_path(sentence_id)

        self.record_button.setEnabled(False)
        self.ai_button.setEnabled(False)
        self.latest_result = None
        self.latest_ai_coaching = None
        self.ai_coaching_text.setText(format_ai_coaching(None))
        self.recognized_label.setText("인식된 문장: -")
        self.target_label.setText(f"정답 문장: {self.prompt_label.text()}")
        self.status_label.setText(f"음성을 감지해 자동 녹음한 뒤 Whisper {DEFAULT_UI_ASR_MODEL_SIZE}로 인식합니다...")

        self.worker = EvaluationWorker(sentence_id, output_path, self.manifest_path)
        self.worker.completed.connect(self._show_result)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _show_result(self, result: EvaluationResult, output_path: Path) -> None:
        self.latest_result = result
        self.score_value.setText(f"{result.score:.1f}")
        self.text_score_value.setText(f"{result.text_score:.1f}")
        self.status_label.setText(f"평가 완료: {output_path}")
        self.recognized_label.setText(f"인식된 문장: {result.recognized_text or '-'}")
        self.target_label.setText(f"정답 문장: {result.closest_text}")
        self.ai_button.setEnabled(True)
        self.start_ai_coaching()

    def start_ai_coaching(self) -> None:
        if self.latest_result is None:
            self.status_label.setText("먼저 녹음 평가를 실행해 주세요.")
            return

        self.ai_button.setEnabled(False)
        self.status_label.setText("코칭을 생성하는 중입니다...")
        self.ai_coaching_text.setText(format_ai_coaching("AI 코칭을 생성하는 중입니다..."))

        self.ai_worker = AiCoachWorker(self.latest_result)
        self.ai_worker.completed.connect(self._show_ai_coaching)
        self.ai_worker.failed.connect(self._show_ai_error)
        self.ai_worker.finished.connect(self._ai_worker_finished)
        self.ai_worker.start()

    def _show_ai_coaching(self, coaching: str) -> None:
        self.latest_ai_coaching = coaching
        self.status_label.setText("AI 코칭 생성 완료")
        self.ai_coaching_text.setText(format_ai_coaching(coaching))

    def _show_ai_error(self, message: str) -> None:
        message = clean_model_text(message)
        self.status_label.setText("AI 코칭을 생성하지 못했습니다.")
        self.ai_coaching_text.setText(format_ai_coaching(message))
        QMessageBox.warning(self, "AI 코칭 오류", message)

    def start_conversation_prompt(self) -> None:
        self.new_conversation_button.setEnabled(False)
        self.answer_conversation_button.setEnabled(False)
        self.current_conversation_prompt = None
        self.conversation_prompt_label.setText("질문을 준비하는 중입니다...")
        self.conversation_recognized_label.setText("내 답변: -")
        self.conversation_feedback_text.setText(format_conversation_feedback(None))
        self.status_label.setText("대화 질문을 준비하는 중입니다...")

        self.conversation_prompt_worker = ConversationPromptWorker(self.conversation_prompt_history)
        self.conversation_prompt_worker.completed.connect(self._show_conversation_prompt)
        self.conversation_prompt_worker.failed.connect(self._show_conversation_error)
        self.conversation_prompt_worker.finished.connect(self._conversation_prompt_worker_finished)
        self.conversation_prompt_worker.start()

    def _show_conversation_prompt(self, prompt: str) -> None:
        prompt = clean_model_text(prompt)
        self.current_conversation_prompt = prompt
        if prompt and prompt not in self.conversation_prompt_history:
            self.conversation_prompt_history.append(prompt)
        self.conversation_prompt_label.setText(prompt)
        self.answer_conversation_button.setEnabled(True)
        self.status_label.setText("질문 준비 완료")

    def start_conversation_answer(self) -> None:
        if not self.current_conversation_prompt:
            self.status_label.setText("먼저 질문을 받아 주세요.")
            return

        self.new_conversation_button.setEnabled(False)
        self.answer_conversation_button.setEnabled(False)
        self.conversation_recognized_label.setText("내 답변: 녹음 및 인식 중...")
        self.conversation_feedback_text.setText(format_conversation_feedback("대화 평가를 생성하는 중입니다..."))
        self.status_label.setText(f"답변을 녹음한 뒤 Whisper {DEFAULT_UI_ASR_MODEL_SIZE}로 인식합니다...")

        self.conversation_response_worker = ConversationResponseWorker(
            self.current_conversation_prompt,
            build_conversation_recording_path(),
        )
        self.conversation_response_worker.completed.connect(self._show_conversation_feedback)
        self.conversation_response_worker.failed.connect(self._show_conversation_error)
        self.conversation_response_worker.finished.connect(self._conversation_response_worker_finished)
        self.conversation_response_worker.start()

    def _show_conversation_feedback(self, recognized_text: str, feedback: str) -> None:
        self.conversation_recognized_label.setText(f"내 답변: {recognized_text or '-'}")
        self.conversation_feedback_text.setText(format_conversation_feedback(feedback))
        self.status_label.setText("대화 평가 완료")

    def _show_conversation_error(self, message: str) -> None:
        message = clean_model_text(message)
        self.status_label.setText("대화 연습 중 오류가 발생했습니다.")
        self.conversation_feedback_text.setText(format_conversation_feedback(message))
        QMessageBox.warning(self, "대화 연습 오류", message)

    def _show_error(self, message: str) -> None:
        message = clean_model_text(message)
        self.status_label.setText("평가 중 오류가 발생했습니다.")
        self.ai_coaching_text.setText(format_ai_coaching(message))
        QMessageBox.critical(self, "실행 오류", message)

    def _worker_finished(self) -> None:
        self.record_button.setEnabled(True)
        self.worker = None

    def _ai_worker_finished(self) -> None:
        self.ai_button.setEnabled(self.latest_result is not None)
        self.ai_worker = None

    def _conversation_prompt_worker_finished(self) -> None:
        self.new_conversation_button.setEnabled(True)
        self.conversation_prompt_worker = None

    def _conversation_response_worker_finished(self) -> None:
        self.new_conversation_button.setEnabled(True)
        self.answer_conversation_button.setEnabled(self.current_conversation_prompt is not None)
        self.conversation_response_worker = None


def _check_runtime() -> None:
    manifest_path = choose_reference_manifest()
    options = load_sentence_options(manifest_path)
    print("ui_runtime_ok")
    print(f"reference_manifest={manifest_path}")
    print(f"sentence_count={len(options)}")
    print(f"default_asr_model={DEFAULT_UI_ASR_MODEL_SIZE}")
    print(f"first_sentence={options[0][0]} | {options[0][1]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the local pronunciation desktop UI.")
    parser.add_argument("--check", action="store_true", help="verify UI dependencies without opening a window")
    args = parser.parse_args(argv)

    if args.check:
        _check_runtime()
        return 0

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = PronunciationWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
