import sys
from datetime import date

import fitz
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtWidgets import QLabel, QShortcut, QScrollArea, QMainWindow, QApplication
from qfluentwidgets import ScrollArea
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Image, Table
from reportlab.platypus import Spacer

from Users import UserManager
from utils.myutil import get_video_info, Globals
import common.resource

# 页面大小
PAGE_HEIGHT = A4[1]
PAGE_WIDTH = A4[0]
# 注册字体
song = "song"
pdfmetrics.registerFont(TTFont('song', 'STSONG.ttf'))


class PDF:

    def __init__(self):
        self.pdf_data = []
        # 获取样例样式表
        style = getSampleStyleSheet()

        # 设置一级标题样式
        ts = style['Heading1']
        ts.fontName = 'song'  # 字体名
        ts.fontSize = 25  # 字体大小
        ts.leading = 30  # 行间距
        ts.alignment = 1  # 居中
        ts.bold = True
        self.Level1Style = ts

        # 设置二级标题样式
        hs = style['Heading2']
        hs.fontName = 'song'  # 字体名
        hs.fontSize = 15  # 字体大小
        hs.leading = 20  # 行间距
        hs.textColor = colors.black  # 字体颜色
        hs.spaceBefore = 10
        hs.spaceAfter = 10
        hs.bold = True
        self.Level2Style = hs

        # 设置普通文本样式
        ns = style['Normal']
        ns.fontName = 'song'
        ns.fontSize = 12
        ns.alignment = 4  # 两端对齐
        ns.firstLineIndent = ns.fontSize * 2  # 第一行开头空格
        ns.leading = 20
        self.bodyTextStyle = ns

    @staticmethod
    # 绘制文本段落
    def draw_text(st, text: str):
        return Paragraph(text, st)

    @staticmethod
    # 绘制图片
    def draw_img(path, width=None):
        img = Image(path)  # 读取图片
        # 计算宽高比
        implicit = img.drawWidth / img.drawHeight
        if width:
            img.drawWidth = width * cm
        else:
            img.drawWidth = 6 * cm
        img.drawHeight = img.drawWidth / implicit
        return img

    @staticmethod
    # 绘制表格
    def draw_table(args, width=None, image_path=None):
        # 列宽度设置为120像素
        col_width = width if width else 50
        # 定义表格样式
        style = [
            ('FONTNAME', (0, 0), (-1, -1), 'song'),  # 使用宋体作为表格字体
            ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行字体大小为12号
            ('FONTSIZE', (0, 1), (-1, -1), 10),  # 从第二行开始，字体大小为10号
            ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),  # 第一行背景颜色为淡蓝色
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 表格内所有单元格水平居中
            # ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # 从第二行开始，单元格左对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有单元格垂直居中对齐
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),  # 表格内文字颜色为暗石板灰色
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 表格网格线颜色为灰色，线宽0.5
            # ('SPAN', (0, 1), (2, 1)),  # 合并第二行的三个单元格
        ]

        table_data = args
        # 检查图片路径是否提供
        if image_path:
            # 准备表格数据，将图片添加到每一行的第一个单元格
            table_data = []
            index = 0
            for row in args:
                if row[0] == '异常图片':
                    table_data.append(row)
                    continue
                # 创建图片对象
                img = Image(image_path[index])
                index += 1
                img.hAlign = 'LEFT'  # 图片水平左对齐
                img.vAlign = 'CENTER'  # 图片垂直居中对齐
                # 计算宽高比
                implicit = img.drawWidth / img.drawHeight
                img.drawWidth = 40
                img.drawHeight = img.drawWidth / implicit
                table_data.append([img] + row)

        table = Table(table_data, colWidths=col_width, style=style)  # 根据传入的参数和样式创建表格

        return table  # 返回创建的表格对象

    @staticmethod
    # 绘制条形图
    def draw_bar(bar_data: list, ax: list, items: list):
        drawing = Drawing(500, 200)
        bc = VerticalBarChart()
        # 设置图表位置和大小
        bc.x = 45
        bc.y = 45
        bc.height = 150
        bc.width = 350
        bc.data = bar_data
        bc.strokeColor = colors.black  # 设置轴线颜色
        # 设置y轴刻度
        bc.valueAxis.valueMin = 0
        bc.valueAxis.valueMax = 20
        bc.valueAxis.valueStep = 5
        # 设置x轴标签属性
        bc.categoryAxis.labels.dx = 2
        bc.categoryAxis.labels.dy = -8
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.labels.fontName = 'song'
        bc.categoryAxis.categoryNames = ax

        # 添加图例
        leg = Legend()
        leg.fontName = 'song'
        leg.alignment = 'right'
        leg.boxAnchor = 'ne'
        leg.x = 475
        leg.y = 140
        leg.dxTextSpace = 10
        leg.columnMaximum = 3
        leg.colorNamePairs = items
        drawing.add(leg)
        drawing.add(bc)
        return drawing

    @staticmethod
    def draw_page_number(c, doc):
        """
        绘制PDF文件的页码。
        """
        # 设置边框颜色
        c.setStrokeColor(colors.dimgrey)
        # 绘制线条
        c.line(30, PAGE_HEIGHT - 790, 570, PAGE_HEIGHT - 790)
        # 绘制页脚文字
        c.setFont(song, 8)
        c.setFillColor(colors.black)
        # 右侧
        c.drawString(30, PAGE_HEIGHT - 810, f"生成日期：{date.today().isoformat()}")
        # 居中
        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 810, f"第{doc.page}页")

    @staticmethod
    def draw_page_header(c, doc):
        """
        绘制PDF文件的页眉。
        """
        # 设置边框颜色
        c.setStrokeColor(colors.dimgrey)
        # 绘制线条
        c.setLineWidth(2)
        c.line(85, 815, PAGE_WIDTH - 30, 815)
        # 绘制页眉文字
        c.setFont(song, 8)
        c.setFillColor(colors.black)
        # 左侧
        username = Globals.user.username if Globals.user else '游客'
        c.drawString(85, 800,
                     f"e视平安\t识别完毕，共花费{Globals.Identify_use_time}\t用户:{username}\t生成日期：{date.today().isoformat()}")
        # 绘制图片
        c.drawImage("resources/UI/logo_new1.png", 30, 780, 50, 50)

    def onFirstPage(self, c: Canvas, doc):
        # 绘制页脚
        self.draw_page_number(c, doc)
        self.draw_page_header(c, doc)

    def onLaterPages(self, c: Canvas, doc):
        # 绘制页眉
        self.draw_page_number(c, doc)  # 在页眉位置绘制当前页码

    def build_pdf(self, pdf_save_path, source=None, save_path=None, model_select=None, select_labels=None,
                  UseCam=False):
        # 初始化内容列表
        content = []

        # 添加标题和图片
        content.append(self.draw_text(self.Level1Style, '识别报告'))
        content.append(Spacer(1, 0.5 * cm))
        content.append(self.draw_img("resources/UI/logo_new1.png", width=4))
        content.append(Spacer(1, 0.5 * cm))

        # 报告概述
        content.append(self.draw_text(self.Level2Style, '报告概述'))
        text = ('报告概述本报告旨在总结通过先进的视频分析技术（YOLO和SlowFast模型）对用户上传的视频进行异常行为识别的结果。'
                '报告提供了详细的识别结果、分析和建议，以便用户能够理解视频中发生的异常事件，并采取相应的后续行动。')
        content.append(self.draw_text(self.bodyTextStyle, text))

        # 基本信息
        content.append(self.draw_text(self.Level2Style, '识别基本信息'))
        if not UseCam:
            total_frames, frame_rate, hours, minutes, seconds, file_size, formatted_date, width, height = get_video_info(
                source)
            texts = [
                f'视频名称：{source}',
                f'视频帧数：{total_frames}',
                f'视频时长：{hours}:{minutes}:{seconds}',
                f'视频帧率：{frame_rate}',
                f'视频分辨率：{width}x{height}',
                f'修改日期：{formatted_date}',
                f'视频大小：{file_size}',
                f'识别结果视频保存位置：{save_path}',
                f'识别模型：{model_select}',
                f"识别标签集：{select_labels}"
            ]
        else:
            texts = [
                f'视频名称：{source}',
                f'识别结果视频保存位置：{save_path}',
                f'识别模型：{model_select}',
                f"识别标签集：{select_labels}"
            ]
        for text in texts:
            content.append(self.draw_text(self.bodyTextStyle, text))

        # 识别方法
        content.append(self.draw_text(self.Level2Style, '识别方法'))
        text = ('本次识别任务采用了两种深度学习模型：YOLO和SlowFast。这两种模型均经过特别训练，以识别视频中的异常行为。'
                'YOLO系列模型以其高速和准确性而闻名，而SlowFast模型则专注于捕捉视频中的快速动作和慢速变化。')
        content.append(self.draw_text(self.bodyTextStyle, text))

        # 异常行为列表
        content.append(self.draw_text(self.Level2Style, '识别列表'))
        content.append(self.draw_text(self.bodyTextStyle, '本次识别共发现异常行为，如下：'))
        data = [['异常图片', '秒数', '编号', '类别', '动作', '坐标', '时间']]
        img_list = [data[0] for data in self.pdf_data]
        self.pdf_data = [data[1:] for data in self.pdf_data]
        for k in self.pdf_data:
            data.append(k)
        content.append(self.draw_table(data, width=[70, 40, 40, 40, 120, 110, 120], image_path=img_list))

        # 折线图
        content.append(self.draw_text(self.Level2Style, '折线图统计'))
        content.append(self.draw_img('result/LineChart.png', width=10))
        # 饼图
        content.append(self.draw_text(self.Level2Style, '饼图统计'))
        content.append(self.draw_img('result/PieChart.png', width=10))

        # 总结和建议
        content.append(self.draw_text(self.Level2Style, '总结和建议'))
        exp_list = []
        for data in self.pdf_data:
            if data[2] not in exp_list:
                exp_list.append(data[2])
        texts = (
            f'本次识别结果共发现{len(exp_list)}个异常行为。',
            '根据本次识别结果，建议用户采取以下后续行动',
            '1. 对识别到的异常行为进行评估和确认。',
            '2. 根据评估结果，采取相应的后续行动，例如通知相关人员、协调处理等。',
            '3. 持续关注和优化识别模型，以提高识别准确率。',
        )
        for text in texts:
            content.append(self.draw_text(self.bodyTextStyle, text))

        # 创建文档实例
        doc = SimpleDocTemplate(pdf_save_path, pagesize=A4)

        # 设置页边距
        # doc.leftMargin = 30
        # doc.rightMargin = 30
        doc.bottomMargin = 50
        doc.topMargin = 50

        # 构建文档内容
        doc.build(content, onFirstPage=self.onFirstPage, onLaterPages=self.onLaterPages)

        print('PDF文件已生成')

        # # 添加二级标题
        # content.append(self.draw_text(self.Level2Style, '游戏厂商统计'))
        # # 添加图表
        # b_data = [(2, 4, 6, 12, 8, 16), (12, 14, 17, 9, 12, 7)]
        # ax_data = ['任天堂', '南梦宫', '科乐美', '卡普空', '世嘉', 'SNK']
        # leg_items = [(colors.red, '街机'), (colors.green, '家用机')]
        # content.append(self.draw_bar(b_data, ax_data, leg_items))
        # content.append(self.draw_bar(b_data, ax_data, leg_items))
        # content.append(self.draw_bar(b_data, ax_data, leg_items))


class MyQScrollArea(QScrollArea):
    change_page = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # 如果同时按下了 Ctrl 键，忽略滚轮事件的默认处理
            event.ignore()
        else:
            # 否则，允许滚轮事件的默认处理
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:  # 检查是否按下上/下箭头键
            if self.verticalScrollBar().value() == 0:
                self.change_page.emit(False)
        elif event.key() == Qt.Key_Down:
            if self.verticalScrollBar().value() == self.verticalScrollBar().maximum():
                self.change_page.emit(True)
        # 其他按键事件，交由父类处理
        super().keyPressEvent(event)


class PDFReader(QMainWindow):
    def __init__(self, pdf_path):
        super().__init__()
        self.path = pdf_path
        self.doc = fitz.open(self.path)
        self.page_count = self.doc.page_count
        self.current_page = 0
        self.zoom_level = QApplication.instance().devicePixelRatio()

        # 创建一个标签来显示PDF页面
        self.label = QLabel()
        self.scrollArea = MyQScrollArea(self)
        self.scrollArea.change_page.connect(self.turn_page)
        self.scrollArea.setWidget(self.label)
        self.scrollArea.setAlignment(Qt.AlignCenter)
        self.scrollBar = self.scrollArea.verticalScrollBar()
        self.label.setScaledContents(True)
        self.update_page_display(True)
        self.setCentralWidget(self.scrollArea)
        self.setup_shortcuts()

    @staticmethod
    def render_page(page):
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        return pm

    def setup_shortcuts(self):
        zoom_in = QShortcut("Ctrl++", self)
        zoom_in.activated.connect(lambda: self.zoom_page(True))

        zoom_out = QShortcut("Ctrl+-", self)
        zoom_out.activated.connect(lambda: self.zoom_page(False))

    def wheelEvent(self, event):
        # 检查是否按住了 Ctrl 键
        if event.modifiers() & Qt.ControlModifier:
            # 计算缩放比例
            zoom_in = event.angleDelta().y() > 0
            # 应用缩放
            self.zoom_page(zoom_in)

    def turn_page(self, is_next):
        if is_next:
            self.current_page += 1
        else:
            self.current_page -= 1
        self.current_page = (self.current_page + len(self.doc)) % len(self.doc)
        self.update_page_display(is_next)

    def zoom_page(self, plus):
        self.zoom_level += 0.1 if plus else -0.1
        if self.zoom_level < 0.1:
            self.zoom_level = 0.1
        self.update_page_display(is_zoom=True)

    def update_page_display(self, is_up=None, is_zoom=None):
        # 计算新缩放矩阵
        trans_a = int(self.zoom_level * 100)  # 将zoom_level转换为百分比
        trans_b = trans_a  # 这里假设水平和垂直缩放相同
        trans = fitz.Matrix(trans_a / 100, trans_b / 100).prerotate(0)

        # 获取当前页面的像素映射
        pix = self.doc[0].get_pixmap(matrix=trans)
        page_width, page_height = pix.width, pix.height
        # 确定图像格式
        fmt = QImage.Format_RGBA8888 if pix.alpha else QImage.Format_RGB888
        total_height = page_height * self.page_count + 5 * (self.page_count - 1)
        # 创建存放合并后的图像
        merged_image = QImage(page_width, total_height, fmt)
        painter = QPainter(merged_image)
        y_offset = 0
        # 绘制每一页的图像
        for page_num in range(self.doc.page_count):
            current_page = self.doc[page_num]
            pix = current_page.get_pixmap(matrix=trans)
            page_image = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
            painter.drawImage(0, y_offset, page_image)
            y_offset += pix.height + 5
        painter.end()
        # 创建QPixmap对象并绘制
        pixmap = QPixmap.fromImage(merged_image)
        self.label.setPixmap(pixmap)
        # 调整QLabel的大小以适应新的pixmap尺寸
        self.label.resize(pixmap.size() / QApplication.instance().devicePixelRatio())

        if is_up is None or is_zoom is True:
            return
        if is_up:
            # 将滚轮设置到顶部
            self.scrollBar.setValue(0)
        else:
            # 滚动到底部
            self.scrollBar.setValue(self.scrollBar.maximum())
        # 更新滚动区域的布局
        self.scrollArea.updateGeometry()


def pdf_reader(path):
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    pdf_reader = PDFReader(path)
    pdf_reader.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    pdf_reader('D:/0000/sadvs/result/result.pdf')
