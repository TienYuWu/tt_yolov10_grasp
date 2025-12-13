"""
Output Console - 檢測結果日誌顯示控制台

提供統一的日誌管理界面，支援多種訊息類型的彩色編碼和自動捲動
"""

from datetime import datetime
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


class OutputConsole(QTableWidget):
    """檢測輸出日誌控制台 - 基於 QTableWidget 的日誌顯示元件

    特性：
    - 3 欄位設計：Time (時間戳)、Type (訊息類型)、Message (訊息內容)
    - 支援 4 種訊息類型：Info、Error、Result、Warning
    - 彩色編碼和自動排版
    - 深色主題樣式
    - 自動捲動到最新訊息

    使用方式：
        console = OutputConsole()
        console.log_info("系統初始化完成")
        console.log_result("檢測完成，發現 5 個物體")
        console.log_error("模型載入失敗")
    """

    # 訊息類型常數
    MSG_INFO = "Info"
    MSG_ERROR = "Error"
    MSG_RESULT = "Result"
    MSG_WARNING = "Warning"

    # 訊息類型對應的顏色 (深色主題)
    TYPE_COLORS = {
        MSG_INFO: "#4ec9b0",      # 青綠色
        MSG_ERROR: "#f48771",     # 紅色
        MSG_RESULT: "#dcdcaa",    # 黃色
        MSG_WARNING: "#ce9178"    # 橙色
    }

    # 日誌最大條目數（超過時移除最舊項）
    MAX_LOG_ENTRIES = 1000

    def __init__(self, parent=None):
        """初始化 Output Console

        Args:
            parent: 父 Widget
        """
        super().__init__(parent)
        self._init_ui()
        self._apply_dark_theme()

    def _init_ui(self):
        """初始化 UI 元素"""
        # 設定 3 欄位
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Time", "Type", "Message"])

        # 欄位寬度設定
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.setColumnWidth(0, 120)  # Time 欄位
        self.setColumnWidth(1, 80)   # Type 欄位

        # 隱藏垂直標頭 (行號)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)

        # 選取模式
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)

        # 禁止編輯
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # 禁用排序 (避免自動打亂日誌順序)
        self.setSortingEnabled(False)

        # 設定行高
        self.verticalHeader().setDefaultSectionSize(28)

    def _apply_dark_theme(self):
        """應用深色主題樣式"""
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
                gridline-color: #3e3e42;
                border: 1px solid #3e3e42;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                alternate-background-color: #252526;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QTableWidget::item:hover {
                background-color: #2d2d30;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #d4d4d4;
                padding: 6px;
                border: 1px solid #3e3e42;
                font-weight: bold;
                text-align: left;
            }
            QHeaderView::section:hover {
                background-color: #3e3e42;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #3e3e42;
                min-height: 20px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #505050;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 12px;
                margin: 0px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background-color: #3e3e42;
                min-width: 20px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #505050;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

    def log_message(self, msg_type: str, message: str):
        """添加一條日誌訊息

        Args:
            msg_type: 訊息類型 (Info/Error/Result/Warning)
            message: 訊息內容

        Examples:
            console.log_message(OutputConsole.MSG_INFO, "系統就緒")
            console.log_message(OutputConsole.MSG_ERROR, "載入失敗")
        """
        # 獲取當前時間 (精確到毫秒)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 插入新行到底部
        row_position = self.rowCount()
        self.insertRow(row_position)

        # ============ Time 欄位 ============
        # 灰色、居中對齐
        time_item = QTableWidgetItem(timestamp)
        time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        time_item.setForeground(QColor("#808080"))
        self.setItem(row_position, 0, time_item)

        # ============ Type 欄位 ============
        # 彩色編碼、居中對齐
        type_item = QTableWidgetItem(msg_type)
        type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)

        # 根據類型設定顏色
        color = self.TYPE_COLORS.get(msg_type, "#d4d4d4")
        type_item.setForeground(QColor(color))

        # Error 類型加粗
        if msg_type == self.MSG_ERROR:
            font = type_item.font()
            font.setBold(True)
            type_item.setFont(font)

        self.setItem(row_position, 1, type_item)

        # ============ Message 欄位 ============
        # 左對齐、自動換行
        message_item = QTableWidgetItem(message)
        message_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        message_item.setToolTip(message)  # 長訊息可透過 Tooltip 查看
        self.setItem(row_position, 2, message_item)

        # ============ 限制日誌條目 ============
        # 防止日誌過多導致性能問題
        if self.rowCount() > self.MAX_LOG_ENTRIES:
            self.removeRow(0)  # 移除最舊的條目

        # ============ 自動捲動到底部 ============
        self.scrollToBottom()

    def clear_logs(self):
        """清空所有日誌"""
        self.setRowCount(0)

    # ============ 快捷方法 ============
    def log_info(self, message: str):
        """記錄 Info 類型訊息

        Args:
            message: 訊息內容

        Example:
            console.log_info("檢測開始")
        """
        self.log_message(self.MSG_INFO, message)

    def log_error(self, message: str):
        """記錄 Error 類型訊息

        Args:
            message: 訊息內容

        Example:
            console.log_error("模型載入失敗: 檔案不存在")
        """
        self.log_message(self.MSG_ERROR, message)

    def log_result(self, message: str):
        """記錄 Result 類型訊息

        Args:
            message: 訊息內容

        Example:
            console.log_result("檢測完成，發現 5 個物體")
        """
        self.log_message(self.MSG_RESULT, message)

    def log_warning(self, message: str):
        """記錄 Warning 類型訊息

        Args:
            message: 訊息內容

        Example:
            console.log_warning("相機幀率低於預期")
        """
        self.log_message(self.MSG_WARNING, message)

    def export_logs(self, filepath: str):
        """匯出日誌到文字檔

        Args:
            filepath: 匯出路徑 (如: "output.txt")

        Returns:
            bool: 匯出是否成功

        Example:
            if console.export_logs("logs/detection_20250211.txt"):
                print("日誌已匯出")
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for row in range(self.rowCount()):
                    time_text = self.item(row, 0).text()
                    type_text = self.item(row, 1).text()
                    msg_text = self.item(row, 2).text()
                    f.write(f"[{time_text}] [{type_text:8}] {msg_text}\n")

            self.log_result(f"日誌已匯出至: {filepath}")
            return True

        except Exception as e:
            self.log_error(f"匯出日誌失敗: {str(e)}")
            return False
