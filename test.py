import sys
import vtkmodules.all as vtk
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox, QDial, QLabel, QMenuBar, QFileDialog, QGridLayout
from PySide6.QtWidgets import QLineEdit, QPushButton, QMessageBox,  QTableWidget, QTableWidgetItem, QDialog, QVBoxLayout, QTextEdit, QMenu, QSlider, QDoubleSpinBox
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QImage, QPainter, QRegion, QMouseEvent, QPixmap, QPen, QCursor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import pydicom
import numpy as np
from vtkmodules.util import numpy_support
import itk
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
import pandas as pd
from collections import deque


class MouseInteractorStyle(vtk.vtkInteractorStyleImage):
    def __init__(self, renderer, image_actor):
        super().__init__()
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.renderer = renderer
        self.image_actor = image_actor
        self.dragging = False
        self.last_mouse_position = None

    def left_button_press_event(self, obj, event):
        self.last_mouse_position = self.GetInteractor().GetEventPosition()
        print(self.last_mouse_position)
        self.dragging = True
        self.OnLeftButtonDown()
        return

    def left_button_release_event(self, obj, event):
        self.dragging = False
        self.OnLeftButtonUp()
        return

    def mouse_move_event(self, obj, event):
        if self.dragging and self.image_actor:
            mouse_position = self.GetInteractor().GetEventPosition()
            dx = mouse_position[0] - self.last_mouse_position[0]
            dy = mouse_position[1] - self.last_mouse_position[1]
            self.last_mouse_position = mouse_position

            current_position = self.image_actor.GetPosition()
            self.image_actor.SetPosition(current_position[0] + dx, current_position[1] + dy, 0)
            self.GetInteractor().GetRenderWindow().Render()

        self.OnMouseMove()
        return


class StateSnapshot:
    def __init__(self, x, y, z, rotate_x, rotate_y, rotate_z):
        self.x = x
        self.y = y
        self.z = z
        self.rotate_x = rotate_x
        self.rotate_y = rotate_y
        self.rotate_z = rotate_z


def itk_to_vtk_image(itk_image):
    itk_array = itk.GetArrayViewFromImage(itk_image)
    vtk_image = vtk.vtkImageData()
    depth, height, width = itk_array.shape
    vtk_image.SetDimensions(width, height, depth)
    vtk_image.AllocateScalars(vtk.VTK_SHORT, 1)
    vtk_data_array = numpy_support.numpy_to_vtk(itk_array.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    return vtk_image


class DICOMViewer:
    def __init__(self, dicom_file=None):
        self.dicom_file = dicom_file
        self.vtk_image = None
        self.slice_thickness = None
        self.pixel_spacing = None

        self.system = 0

        # 关键点
        self.AODA = None
        self.ANS = None
        self.HtR = None
        self.HtL = None
        self.SR = None

        self.marked_points = []
        self.key_points = []
        self.distances = []
        self.angles = []
        self.markers = []  # 储存现在坐标在三视图上的红点actor
        self.lines = []  # 储存线条actor
        self.state_snapshots = deque(maxlen=3)  # 保存最近的三次状态快照

        self.x_last = 0
        self.y_last = 0
        self.z_last = 0
        self.x_angle_last = 0
        self.y_angle_last = 0
        self.z_angle_last = 0

        self.lr_count = 0
        self.fh_count = 0
        self.tb_count = 0

        self.width = 0
        self.height = 0
        self.depth = 0

        if dicom_file:
            self.vtk_image=self.load_dicom_files(dicom_file)

    def load_dicom_files(self, filenames):
        reader = itk.ImageSeriesReader[itk.Image[itk.SS, 3]].New()
        dicom_io = itk.GDCMImageIO.New()
        reader.SetImageIO(dicom_io)
        reader.SetFileNames(filenames)

        try:
            reader.Update()
        except Exception as e:
            print(f"Error reading DICOM files: {e}")
            return None

        itk_image = reader.GetOutput()
        vtk_image = itk_to_vtk_image(itk_image)

        dicom_data = pydicom.dcmread(filenames[0])
        self.slice_thickness = dicom_data.SliceThickness
        self.pixel_spacing = dicom_data.PixelSpacing

        dimensions = vtk_image.GetDimensions()
        self.width, self.height, self.depth = dimensions

        del reader
        del dicom_io
        del dicom_data
        return vtk_image


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.lr_count = 0
        self.fh_count = 0
        self.tb_count = 0

        self.head = True
        self.slice_thickness = None
        self.dicom_viewers = []  # 用于存储加载的 DICOMViewer 实例
        self.current_viewer_index = None

        self.xz_plane_3d_actor = None
        self.yz_plane_3d_actor = None
        self.xy_plane_3d_actor = None

        self.system = 0
        self.euler_angles = [0] * 3
        self.euler_angles_map = [0] * 3  # 记录为了使得图像正常显示而设置的图像欧拉角
        self.origin_physical = [0] * 3  # 欧拉角转出的SR
        self.origin_physical_map = [0] * 3  # 图片欧拉角转出的SR
        self.origin_world = [0] * 3  # 记录原始的SR点坐标
        self.center = [0] * 3  # 记录图像的中心点坐标
        self.setWindowTitle("DICOM Viewer with 3D Reconstruction and Slices")

        # 初始化变量，控制监听器是否监听任务。
        self.picking = False  # 开启三视图点击联动
        self.measuring = False  # 测距离
        self.measuring_angle = False  # 测角度
        self.measuring_horizontal_angle = False  # 测水平角度
        self.marking = False  # 是否标记
        self.erasing = False  # 擦除
        self.projection_3d = False  # 是否进行3D映射
        self.projection_3d_2d = False  # 是否开启3D反射回2D
        self.coords_plane_display = False  # 是否展示坐标平面
        self.first_open = True

        self.erase_radius = 20.0  # 初始化擦除半径
        self.marker_radius = 3.0  # 初始化标记红点大小（半径）

        # 3D映射
        self.slice_marker = None  # 现在所处点的marker
        self.x_line_actor = None  # x轴
        self.y_line_actor = None  # y轴
        self.z_line_actor = None  # z轴

        self.flip = False

        # 关键点
        self.AODA = None
        self.ANS = None
        self.HtR = None
        self.HtL = None
        self.SR = None

        self.marked_points = []
        self.key_points = []
        self.distances = []
        self.angles = []
        self.markers = []  # 储存现在坐标在三视图上的红点actor
        self.lines = []  # 储存线条actor
        self.state_snapshots = deque(maxlen=3)  # 保存最近的三次状态快照

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)


        # 创建视图窗口
        self.vtk_widget_axial = QVTKRenderWindowInteractor(self.central_widget)
        self.vtk_widget_coronal = QVTKRenderWindowInteractor(self.central_widget)
        self.vtk_widget_sagittal = QVTKRenderWindowInteractor(self.central_widget)
        self.vtk_widget_3d = QVTKRenderWindowInteractor(self.central_widget)

        # 添加标签
        label_axial = QLabel("Axial Slice (Z)")
        label_coronal = QLabel("Coronal Slice (Y)")
        label_sagittal = QLabel("Sagittal Slice (X)")
        label_3d = QLabel("3D Reconstruction")

        # 配置视图窗口和标签在网格中的位置
        self.layout.addWidget(label_axial, 1, 0)
        self.layout.addWidget(self.vtk_widget_axial, 3, 0, 1, 3)
        self.layout.addWidget(label_coronal, 1, 3)
        self.layout.addWidget(self.vtk_widget_coronal, 3, 3, 1, 3)
        self.layout.addWidget(label_sagittal, 4, 0)
        self.layout.addWidget(self.vtk_widget_sagittal, 6, 0, 1, 3)
        self.layout.addWidget(label_3d, 4, 3)
        self.layout.addWidget(self.vtk_widget_3d, 6, 3, 1, 3)

        # 撤销按钮
        self.undo_button = QPushButton("Undo", self.central_widget)
        self.undo_button.clicked.connect(self.undo)
        self.layout.addWidget(self.undo_button, 0, 0)

        # 初始化渲染窗口和交互器
        self.init_render_windows()

        # 创建菜单和输入字段
        self.create_menu()
        self.create_input_fields()

        # Add coordinate label
        self.coord_label = QLabel("Coordinates: ")
        self.layout.addWidget(self.coord_label, 8, 0)

        self.current_window = 500
        self.current_level = 250

        # 简易操作按钮栏
        self.pick_label = QLabel("Pick point is OFF")
        self.layout.addWidget(self.pick_label, 0, 1)

        self.pick_button = QPushButton("On/Off", self.central_widget)
        self.pick_button.clicked.connect(self.picking_switch)
        self.layout.addWidget(self.pick_button, 0, 2)

        self.flip_LR_button = QPushButton("左右镜像", self.central_widget)
        self.flip_LR_button.clicked.connect(self.flip_LR)
        self.layout.addWidget(self.flip_LR_button, 0, 3)

        self.flip_FH_button = QPushButton("前后镜像", self.central_widget)
        self.flip_FH_button.clicked.connect(self.flip_FH)
        self.layout.addWidget(self.flip_FH_button, 0, 4)

        self.flip_TB_button = QPushButton("上下镜像", self.central_widget)
        self.flip_TB_button.clicked.connect(self.flip_TB)
        self.layout.addWidget(self.flip_TB_button, 0, 5)

        self.set_origin_button = QPushButton("切换", self.central_widget)
        self.set_origin_button.clicked.connect(self.switch_image)
        self.layout.addWidget(self.set_origin_button, 0, 6)

        self.calculate_coordinates_button = QPushButton("关键点建立坐标系", self.central_widget)
        self.calculate_coordinates_button.clicked.connect(self.set_coordinate_system)
        self.layout.addWidget(self.calculate_coordinates_button, 0, 7)

        self.set_SR_button = QPushButton("标记SR", self.central_widget)
        self.set_SR_button.clicked.connect(self.set_SR)
        self.layout.addWidget(self.set_SR_button, 1, 6)

        self.save_SR_button = QPushButton("保存SR", self.central_widget)
        self.save_SR_button.clicked.connect(self.save_SR)
        self.layout.addWidget(self.save_SR_button, 1, 7)

        self.set_AODA_button = QPushButton("标记AODA", self.central_widget)
        self.set_AODA_button.clicked.connect(self.set_AODA)
        self.layout.addWidget(self.set_AODA_button, 2, 6)

        self.save_AODA_button = QPushButton("保存AODA", self.central_widget)
        self.save_AODA_button.clicked.connect(self.save_AODA)
        self.layout.addWidget(self.save_AODA_button, 2, 7)

        self.set_ANS_button = QPushButton("标记ANS", self.central_widget)
        self.set_ANS_button.clicked.connect(self.set_ANS)
        self.layout.addWidget(self.set_ANS_button, 3, 6)

        self.save_ANS_button = QPushButton("保存ANS", self.central_widget)
        self.save_ANS_button.clicked.connect(self.save_ANS)
        self.layout.addWidget(self.save_ANS_button, 3, 7)

        self.set_HtR_button = QPushButton("标记HtR", self.central_widget)
        self.set_HtR_button.clicked.connect(self.set_HtR)
        self.layout.addWidget(self.set_HtR_button, 4, 6)

        self.save_HtR_button = QPushButton("保存HtR", self.central_widget)
        self.save_HtR_button.clicked.connect(self.save_HtR)
        self.layout.addWidget(self.save_HtR_button, 4, 7)

        self.set_HtL_button = QPushButton("标记HtL", self.central_widget)
        self.set_HtL_button.clicked.connect(self.set_HtL)
        self.layout.addWidget(self.set_HtL_button, 5, 6)

        self.save_HtL_button = QPushButton("保存HtL", self.central_widget)
        self.save_HtL_button.clicked.connect(self.save_HtL)
        self.layout.addWidget(self.save_HtL_button, 5, 7)

        self.projection_back_button = QPushButton("3D映射2D", self.central_widget)
        self.projection_back_button.clicked.connect(self.switch_projection_back)
        self.layout.addWidget(self.projection_back_button, 6, 6)

        self.projection_3d_2d_label = QLabel("3映2：关闭")
        self.layout.addWidget(self.projection_3d_2d_label, 6, 7)

        self.show_planes_button = QPushButton("展示坐标平面", self.central_widget)
        self.show_planes_button.clicked.connect(self.coordinate_planes_switch)
        self.layout.addWidget(self.show_planes_button, 7, 6)

        # 添加显示物理位置的标签
        self.physical_position_label = QLineEdit()
        self.physical_position_label.setText("Physical Position: ")
        self.layout.addWidget(self.physical_position_label, 9, 0, 1, 3)

        # 添加显示上次测量角度的标签
        self.last_angle_label = QLineEdit()
        self.last_angle_label.setText("Last Angle: ")
        self.layout.addWidget(self.last_angle_label, 9, 3, 1, 2)

        # 添加显示上次测量距离的标签
        self.last_distance_label = QLineEdit()
        self.last_distance_label.setText("Last Distance: ")
        self.layout.addWidget(self.last_distance_label, 9, 5, 1, 2)

        self.last_h_angle_label = QLineEdit()
        self.last_h_angle_label.setText("Last Horizontal Angle: ")
        self.layout.addWidget(self.last_h_angle_label, 10, 3, 1, 2)

    def create_input_fields(self):
        # 创建和配置输入字段
        self.x_input = QSpinBox(self.central_widget)
        self.y_input = QSpinBox(self.central_widget)
        self.z_input = QSpinBox(self.central_widget)

        self.x_input.setRange(0, 512)
        self.y_input.setRange(0, 512)
        self.z_input.setRange(0, 512)

        # 安排输入字段和标签
        self.layout.addWidget(self.x_input, 5, 0, 1, 1)
        self.layout.addWidget(self.y_input, 2, 3, 1, 1)
        self.layout.addWidget(self.z_input, 2, 0, 1, 1)

        # 连接输入字段的变化事件到视图更新函数
        self.x_input.valueChanged.connect(self.update_views)
        self.y_input.valueChanged.connect(self.update_views)
        self.z_input.valueChanged.connect(self.update_views)

        # 旋转按钮的角度输入框
        self.rotate_x_input = QDoubleSpinBox(self.central_widget)
        self.rotate_y_input = QDoubleSpinBox(self.central_widget)
        self.rotate_z_input = QDoubleSpinBox(self.central_widget)

        self.rotate_x_input.setRange(-1, 360)
        self.rotate_y_input.setRange(-1, 360)
        self.rotate_z_input.setRange(-1, 360)

        self.rotate_x_input.setSingleStep(0.5)
        self.rotate_x_input.setDecimals(2)

        self.rotate_y_input.setSingleStep(0.5)
        self.rotate_y_input.setDecimals(2)

        self.rotate_z_input.setSingleStep(0.5)
        self.rotate_z_input.setDecimals(2)

        self.rotate_x_input.setValue(0)
        self.rotate_x_input.valueChanged.connect(self.update_rotate_x)

        self.rotate_y_input.setValue(0)
        self.rotate_y_input.valueChanged.connect(self.update_rotate_y)

        self.rotate_z_input.setValue(0)
        self.rotate_z_input.valueChanged.connect(self.update_rotate_z)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-1000, 1000)
        self.brightness_slider.setValue(-300)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.layout.addWidget(QLabel("Brightness"), 7, 0)
        self.layout.addWidget(self.brightness_slider, 7, 1)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 4000)
        self.contrast_slider.setValue(2000)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.layout.addWidget(QLabel("Contrast"), 7, 2)
        self.layout.addWidget(self.contrast_slider, 7, 3)

        self.layout.addWidget(self.rotate_x_input, 5, 2)

        self.layout.addWidget(self.rotate_y_input, 2, 5)

        self.layout.addWidget(self.rotate_z_input, 2, 2)

        angle_axial = QLabel("Axial (Z) Angle (degrees):")
        angle_coronal = QLabel("Coronal (Y) Angle (degrees):")
        angle_sagittal = QLabel("Sagittal (X) Angle (degrees):")

        self.layout.addWidget(angle_axial, 1, 2)
        self.layout.addWidget(angle_coronal, 1, 5)
        self.layout.addWidget(angle_sagittal, 4, 2)

    def init_render_windows(self):
        # 获取与该部件关联的渲染窗口
        self.render_window_axial = self.vtk_widget_axial.GetRenderWindow()
        #  获取渲染窗口的交互器，用于处理用户输入（如鼠标和键盘事件）
        self.render_window_interactor_axial = self.render_window_axial.GetInteractor()

        self.render_window_coronal = self.vtk_widget_coronal.GetRenderWindow()
        self.render_window_interactor_coronal = self.render_window_coronal.GetInteractor()

        self.render_window_sagittal = self.vtk_widget_sagittal.GetRenderWindow()
        self.render_window_interactor_sagittal = self.render_window_sagittal.GetInteractor()

        self.render_window_3d = self.vtk_widget_3d.GetRenderWindow()
        self.renderer_3d = vtk.vtkRenderer()
        self.render_window_3d.AddRenderer(self.renderer_3d)
        self.render_window_interactor_3d = self.render_window_3d.GetInteractor()

        # 设置3D窗口的交互样式
        style = vtkInteractorStyleTrackballCamera()
        self.render_window_interactor_3d.SetInteractorStyle(style)

        self.axial_viewer = vtk.vtkResliceImageViewer()
        self.axial_viewer.SetRenderWindow(self.render_window_axial)
        self.axial_viewer.SetupInteractor(self.render_window_interactor_axial)

        self.coronal_viewer = vtk.vtkResliceImageViewer()
        self.coronal_viewer.SetRenderWindow(self.render_window_coronal)
        self.coronal_viewer.SetupInteractor(self.render_window_interactor_coronal)

        self.sagittal_viewer = vtk.vtkResliceImageViewer()
        self.sagittal_viewer.SetRenderWindow(self.render_window_sagittal)
        self.sagittal_viewer.SetupInteractor(self.render_window_interactor_sagittal)

        self.render_window_interactor_axial.SetInteractorStyle(vtk.vtkInteractorStyleImage())
        self.render_window_interactor_coronal.SetInteractorStyle(vtk.vtkInteractorStyleImage())
        self.render_window_interactor_sagittal.SetInteractorStyle(vtk.vtkInteractorStyleImage())

        self.axial_viewer.AddObserver("ModifiedEvent", self.update_inputs_from_viewer)
        self.coronal_viewer.AddObserver("ModifiedEvent", self.update_inputs_from_viewer)
        self.sagittal_viewer.AddObserver("ModifiedEvent", self.update_inputs_from_viewer)

        self.mouse_pressed = False
        self.dragging_line = None
        self.hovered_line = None

    def create_menu(self):
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")

        open_files_action = file_menu.addAction("Open Files")
        open_files_action.triggered.connect(self.open_files)

        open_files_action = file_menu.addAction("Compare Files")
        open_files_action.triggered.connect(lambda: self.compare_window("axial"))

        read_key_points_from_excel_action = file_menu.addAction("从excel中读取关键点")
        read_key_points_from_excel_action.triggered.connect(self.read_key_points_from_excel)

        save_action = file_menu.addAction("Save All")
        save_action.triggered.connect(self.save_screenshot)

        save_axial_action = file_menu.addAction("Save Axial")
        save_axial_action.triggered.connect(self.save_axial_window)

        save_coronal_action = file_menu.addAction("Save coronal")
        save_coronal_action.triggered.connect(self.save_coronal_window)

        save_sagittal_action = file_menu.addAction("Save sagittal")
        save_sagittal_action.triggered.connect(self.save_sagittal_window)

        save_3d_action = file_menu.addAction("Save 3d")
        save_3d_action.triggered.connect(self.save_3d_window)

        operation_menu = menubar.addMenu("Operations")

        mark_action = operation_menu.addAction("标记现在点")
        mark_action.triggered.connect(self.enable_marking)

        measure_simple_action = operation_menu.addAction("测量两点距离")
        measure_simple_action.triggered.connect(self.measure_two_points)

        measure_angle_simple_action = operation_menu.addAction("测量三点角度")
        measure_angle_simple_action.triggered.connect(self.measure_one_angle)

        measure_horizontal_angle_action = operation_menu.addAction("测量水平角")
        measure_horizontal_angle_action.triggered.connect(self.measure_angle_horizontal)

        erase_action = operation_menu.addAction("开启橡皮擦")
        erase_action.triggered.connect(self.enable_erasing)

        disable_erase_action = operation_menu.addAction("关闭橡皮擦")
        disable_erase_action.triggered.connect(self.disable_erasing)

        help_menu = menubar.addMenu("Help")

        commands_action = help_menu.addAction("commands 操作指南")
        commands_action.triggered.connect(self.commands_explained)

        view_menu = menubar.addMenu("View")

        view_marked_points_action = view_menu.addAction("View Marked Points")
        view_marked_points_action.triggered.connect(self.show_marked_points)

        view_key_points_action = view_menu.addAction("View key Points")
        view_key_points_action.triggered.connect(self.show_key_points)

        view_distances_action = view_menu.addAction("View Distances")
        view_distances_action.triggered.connect(self.show_distances)

        view_angles_action = view_menu.addAction("View Angles")
        view_angles_action.triggered.connect(self.show_angles)

        show_slice_position_action = view_menu.addAction("Display Slice Position in 3D")
        show_slice_position_action.triggered.connect(self.show_slice_position_in_3d)

        stop_showing_action = view_menu.addAction("Stop displaying slice position in 3D")
        stop_showing_action.triggered.connect(self.stop_showing_3d)

        self.setMenuBar(menubar)

    def commands_explained(self):
        help_text = (
            "Usage Instructions:\n"
            "\n"
            "1. Open: Load a DICOM file.\n"
            "2. Save: Save the current screenshot of the interface.\n"
            "3. Mark: Mark a specific point in the 3D view.\n"
            "4. Pick Point: Enable picking mode to select points in the slices.\n"
            "5. Use the spin boxes to navigate through different slices.\n"
            "6. The current coordinates are displayed in the input fields and updated as you navigate.\n"
            "画面：鼠标滚轮切换视图，切换后在operations-update下将坐标输入框更新至与视图同步 \n"
            "鼠标处于图像上时，左键长按左右拖拽调整对比度，左键长按上下拖拽调整亮度，右键长按上下拖拽调整图像大小 \n"
        )
        QMessageBox.information(self, "Help", help_text)

    def update_brightness(self, value):
        self.color_level = -value
        self.axial_viewer.SetColorLevel(self.color_level)
        self.coronal_viewer.SetColorLevel(self.color_level)
        self.sagittal_viewer.SetColorLevel(self.color_level)
        self.axial_viewer.Render()
        self.coronal_viewer.Render()
        self.sagittal_viewer.Render()

    def update_contrast(self, value):
        self.color_window = value
        self.axial_viewer.SetColorWindow(self.color_window)
        self.coronal_viewer.SetColorWindow(self.color_window)
        self.sagittal_viewer.SetColorWindow(self.color_window)
        self.axial_viewer.Render()
        self.coronal_viewer.Render()
        self.sagittal_viewer.Render()

    def enable_marking(self):
        self.picking = False  # 三视图点位联动
        self.measuring = False  # 测距离
        self.measuring_angle = False  # 测角度
        self.marking = True  # 是否标记
        self.erasing = False  # 擦除
        self.marking = True
        self.mark_point()

    def mark_point(self):
        if self.marking:
            world_pos = [self.x_input.value(), self.y_input.value(), self.z_input.value()]

            actual_pos = self.add_marker(self.axial_viewer, world_pos)
            print("World: ", world_pos)
            print("Actual: ", actual_pos)
            self.add_marker(self.coronal_viewer, world_pos)
            self.add_marker(self.sagittal_viewer, world_pos)

            point_name = f"Point {len(self.marked_points) + 1}"
            angle = (self.rotate_x_input.value(), self.rotate_y_input.value(), self.rotate_z_input.value())
            physical_pos = self.update_physical_position_label_map(actual_pos[0], actual_pos[1], actual_pos[2])

            self.marked_points.append((point_name, actual_pos, angle, physical_pos))

        self.disable_marking()

    def disable_marking(self):
        # 恢复正常的鼠标样式并停止捕获事件
        # QApplication.restoreOverrideCursor()
        self.marking = False

    def enable_erasing(self):
        self.picking = False  # 三视图点位联动
        self.measuring = False  # 测距离
        self.measuring_angle = False  # 测角度
        self.marking = False  # 是否标记
        self.erasing = True  # 擦除
        self.set_erase_cursor()

    def disable_erasing(self):
        # 恢复正常的鼠标样式并停止捕获事件
        QApplication.restoreOverrideCursor()
        self.erasing = False

    def erase_marker(self, obj, event):
        if self.erasing:
            click_pos = obj.GetEventPosition()
            picker = vtk.vtkCellPicker()
            renderer = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()

            if renderer is not None:
                picker.Pick(click_pos[0], click_pos[1], 0, renderer)
                picked_center = picker.GetPickPosition()
                picked_actors = self.pick_actors_in_radius(renderer, picked_center, self.erase_radius)

                if picked_actors:
                    for picked_actor in picked_actors:
                        if isinstance(picked_actor, vtk.vtkActor) and picked_actor.GetProperty().GetColor() == (
                        1, 0, 0):
                            # 查找并删除表格中对应的点
                            pos = picked_actor.GetCenter()
                            self.marked_points = [mp for mp in self.dicom_viewers[self.current_viewer_index].marked_points if mp[1] != pos]
                        renderer.RemoveActor(picked_actor)
                        obj.GetRenderWindow().Render()

    def pick_actors_in_radius(self, renderer, pick_position, radius):
        picked_actors = []
        actors = renderer.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()

        while actor:
            bounds = actor.GetBounds()
            center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
            distance = np.linalg.norm(np.array(pick_position) - np.array(center))
            if distance <= radius:
                picked_actors.append(actor)
            actor = actors.GetNextActor()

        return picked_actors

    # 设置一个自定义的擦除光标
    def set_erase_cursor(self):
        diameter = self.erase_radius * 2
        pixmap = QPixmap(diameter, diameter)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawEllipse(0, 0, diameter, diameter)
        painter.end()

        cursor = QCursor(pixmap)
        QApplication.setOverrideCursor(cursor)

    def enable_zooming(self):
        self.zoom_activation = True

    def set_SR(self):  # 将目前位置保存为SR
        if self.system == 0:
            current_pos = [self.x_input.value(), self.y_input.value(), self.z_input.value()]
            current_angle = [self.rotate_x_input.value(), self.rotate_y_input.value(), self.rotate_z_input.value()]
            physical_pos = (0, 0, 0)
            new_SR = ("SR", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].SR = new_SR

            self._replace_or_add_key_point(new_SR)
            self.set_SR_button.setStyleSheet("color: green;")
        else:
            current_pos = self.update_world_position_label_map(self.x_input.value(), self.y_input.value(),
                                                               self.z_input.value())
            current_angle = np.array([0, 0, 0])
            physical_pos = self.update_physical_position_label_map(self.x_input.value(), self.y_input.value(),
                                                                   self.z_input.value())
            new_SR = ("SR", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].SR = new_SR

            self._replace_or_add_key_point(new_SR)
            self.set_SR_button.setStyleSheet("color: green;")

    def set_AODA(self):  # 将目前位置保存为AODA
        if self.system == 0:
            current_pos = [self.x_input.value(), self.y_input.value(), self.z_input.value()]
            current_angle = [self.rotate_x_input.value(), self.rotate_y_input.value(), self.rotate_z_input.value()]
            physical_pos = (0, 0, 0)
            new_AODA = ("AODA", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].AODA = new_AODA

            self._replace_or_add_key_point(new_AODA)
            self.set_AODA_button.setStyleSheet("color: green;")
        else:
            current_pos = self.update_world_position_label_map(self.x_input.value(), self.y_input.value(),
                                                               self.z_input.value())
            current_angle = np.array([0, 0, 0])
            physical_pos = self.update_physical_position_label_map(self.x_input.value(), self.y_input.value(),
                                                                   self.z_input.value())
            new_AODA = ("AODA", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].AODA = new_AODA

            self._replace_or_add_key_point(new_AODA)
            self.set_AODA_button.setStyleSheet("color: green;")

    def set_ANS(self):  # 将目前位置保存为ANS
        if self.system == 0:
            current_pos = [self.x_input.value(), self.y_input.value(), self.z_input.value()]
            current_angle = [self.rotate_x_input.value(), self.rotate_y_input.value(), self.rotate_z_input.value()]
            physical_pos = (0, 0, 0)
            new_ANS = ("ANS", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].ANS = new_ANS

            self._replace_or_add_key_point(new_ANS)
            self.set_ANS_button.setStyleSheet("color: green;")
        else:
            current_pos = self.update_world_position_label_map(self.x_input.value(), self.y_input.value(),
                                                               self.z_input.value())
            current_angle = np.array([0, 0, 0])
            physical_pos = self.update_physical_position_label_map(self.x_input.value(), self.y_input.value(),
                                                                   self.z_input.value())
            new_ANS = ("ANS", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].ANS = new_ANS

            self._replace_or_add_key_point(new_ANS)
            self.set_ANS_button.setStyleSheet("color: green;")

    def set_HtR(self):  # 将目前位置保存为HtR
        if self.system == 0:
            current_pos = [self.x_input.value(), self.y_input.value(), self.z_input.value()]
            current_angle = [self.rotate_x_input.value(), self.rotate_y_input.value(), self.rotate_z_input.value()]
            physical_pos = (0, 0, 0)
            new_HtR = ("HtR", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].HtR = new_HtR

            self._replace_or_add_key_point(new_HtR)
            self.set_HtR_button.setStyleSheet("color: green;")
        else:
            current_pos = self.update_world_position_label_map(self.x_input.value(), self.y_input.value(),
                                                               self.z_input.value())
            current_angle = np.array([0, 0, 0])
            physical_pos = self.update_physical_position_label_map(self.x_input.value(), self.y_input.value(),
                                                                   self.z_input.value())
            new_HtR = ("HtR", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].HtR = new_HtR

            self._replace_or_add_key_point(new_HtR)
            self.set_HtR_button.setStyleSheet("color: green;")

    def set_HtL(self):  # 将目前位置保存为HtL
        if self.system == 0:
            current_pos = [self.x_input.value(), self.y_input.value(), self.z_input.value()]
            current_angle = [self.rotate_x_input.value(), self.rotate_y_input.value(), self.rotate_z_input.value()]
            physical_pos = (0, 0, 0)
            new_HtL = ("HtL", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].HtL = new_HtL


            self._replace_or_add_key_point(new_HtL)
            self.set_HtL_button.setStyleSheet("color: green;")
        else:
            current_pos = self.update_world_position_label_map(self.x_input.value(), self.y_input.value(),
                                                               self.z_input.value())
            current_angle = np.array([0, 0, 0])
            physical_pos = self.update_physical_position_label_map(self.x_input.value(), self.y_input.value(),
                                                                   self.z_input.value())
            new_HtL = ("HtL", current_pos, current_angle, physical_pos)

            self.dicom_viewers[self.current_viewer_index].HtL = new_HtL

            self._replace_or_add_key_point(new_HtL)
            self.set_HtL_button.setStyleSheet("color: green;")

    def _replace_or_add_key_point(self, new_point):
        """
        辅助方法：如果关键点已经存在，则替换它；否则添加到列表中。
        同时更新全局变量（如 self.SR）。
        """
        name = new_point[0]  # 获取关键点的名称
        for i, point in enumerate(self.key_points):
            if point[0] == name:  # 如果找到同名关键点
                self.key_points[i] = new_point  # 替换为新值
                self._update_global_variable(new_point)  # 更新全局变量
                return
        self.key_points.append(new_point)  # 如果不存在，则添加到列表
        self._update_global_variable(new_point)  # 更新全局变量


    def _update_global_variable(self, new_point):
        """
        辅助方法：根据关键点名称更新对应的全局变量。
        """
        name = new_point[0]
        if name == "SR":
            self.SR = new_point
        elif name == "AODA":
            self.AODA = new_point
        elif name == "ANS":
            self.ANS = new_point
        elif name == "HtR":
            self.HtR = new_point
        elif name == "HtL":
            self.HtL = new_point

    def jump_to_point(self, point):
        if point:
            pos = point[1]
            angle = point[2]
            self.x_input.setValue(int(pos[0]))
            self.y_input.setValue(int(pos[1]))
            self.z_input.setValue(int(pos[2]))
            self.rotate_x_input.setValue(int(angle[0]))
            self.rotate_y_input.setValue(int(angle[1]))
            self.rotate_z_input.setValue(int(angle[2]))

    def save_SR(self):  # 将SR图像输出
        self.jump_to_point(self.SR)
        self.save_screenshot()

    def save_AODA(self):  # 将AODA图像输出
        self.jump_to_point(self.AODA)
        self.save_screenshot()

    def save_ANS(self):  # 将ANS图像输出
        self.jump_to_point(self.ANS)
        self.save_screenshot()

    def save_HtR(self):  # 将HtR图像输出
        self.jump_to_point(self.HtR)
        self.save_screenshot()

    def save_HtL(self):  # 将HtL图像输出
        self.jump_to_point(self.HtL)
        self.save_screenshot()

    def read_key_points_from_excel(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx)")
        if file_path:
            df = pd.read_excel(file_path)
            df.drop_duplicates(subset=["Name"], keep='last', inplace=True)

            # Update self variables
            self.AODA = self.get_point_from_df(df, "AODA")
            self.ANS = self.get_point_from_df(df, "ANS")
            self.HtL = self.get_point_from_df(df, "HtL")
            self.HtR = self.get_point_from_df(df, "HtR")
            self.SR = self.get_point_from_df(df, "SR")
            self.key_points.clear()
            self.key_points = [self.AODA, self.ANS, self.HtL, self.HtR, self.SR]

            # 在这里更新每一个实例的关键点坐标，用于切换后的直接建坐标系
            self.dicom_viewers[self.current_viewer_index].AODA = self.get_point_from_df(df, "AODA")
            self.dicom_viewers[self.current_viewer_index].ANS = self.get_point_from_df(df, "ANS")
            self.dicom_viewers[self.current_viewer_index].HtL = self.get_point_from_df(df, "HtL")
            self.dicom_viewers[self.current_viewer_index].HtR = self.get_point_from_df(df, "HtR")
            self.dicom_viewers[self.current_viewer_index].SR = self.get_point_from_df(df, "SR")
            self.dicom_viewers[self.current_viewer_index].key_points.clear()
            self.dicom_viewers[self.current_viewer_index].key_points = [self.AODA, self.ANS, self.HtL, self.HtR, self.SR]



    def get_point_from_df(self, df, name):
        row = df[df['Name'] == name]
        if not row.empty:
            coordinates = (row.iloc[0]['X'], row.iloc[0]['Y'], row.iloc[0]['Z'])
            angles = (row.iloc[0]['Angle X'], row.iloc[0]['Angle Y'], row.iloc[0]['Angle Z'])
            physical_coords = (row.iloc[0]['Physical X'], row.iloc[0]['Physical Y'], row.iloc[0]['Physical Z'])
            return (name, coordinates, angles, physical_coords)
        return None


    def show_marked_points(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Marked Points")
        layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z", "Angle_X",
                                         "Angle_Y", "Angle_Z", "Physical X",
                                         "Physical Y", "Physical Z"])
        table.setRowCount(len(self.marked_points))

        for i, (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in enumerate(self.marked_points):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            table.setItem(i, 0, name_item)
            table.setItem(i, 1, QTableWidgetItem(f"{x:.2f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{y:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{z:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{angle_x:.2f}"))
            table.setItem(i, 5, QTableWidgetItem(f"{angle_y:.2f}"))
            table.setItem(i, 6, QTableWidgetItem(f"{angle_z:.2f}"))
            table.setItem(i, 8, QTableWidgetItem(f"{phy_x:.2f}"))
            table.setItem(i, 7, QTableWidgetItem(f"{phy_y:.2f}"))
            table.setItem(i, 9, QTableWidgetItem(f"{phy_z:.2f}"))

        table.cellChanged.connect(lambda row, column: self.save_marked_points(table, row, column))

        layout.addWidget(table)

        # 添加导出按钮
        export_button = QPushButton("Export to Excel", dialog)
        export_button.clicked.connect(self.export_to_excel)
        layout.addWidget(export_button)

        distance_button = QPushButton("显示两点距离", dialog)
        distance_button.clicked.connect(lambda: self.select_two_points(table))
        layout.addWidget(distance_button)

        angle_button = QPushButton("显示三点角度", dialog)
        angle_button.clicked.connect(lambda: self.select_three_points(table))
        layout.addWidget(angle_button)

        angle_horizontal_button = QPushButton("显示水平角度", dialog)
        angle_horizontal_button.clicked.connect(lambda: self.select_horizontal_angle(table))
        layout.addWidget(angle_horizontal_button)

        dialog.setLayout(layout)
        dialog.show()

    def show_key_render_window(self):
        '''
        if not self.key_render_dialog.isVisible():
            self.key_render_dialog.show()
            # 开始交互
            self.key_vtk_widget.Initialize()
            self.key_vtk_widget.Start()
        #self.key_vtk_widget.GetRenderWindow().Render()'''
        self.key_render_dialog = QDialog(self)
        self.key_render_dialog.setWindowTitle("3D Rendering")
        key_render_layout = QVBoxLayout(self.key_render_dialog)

        self.key_vtk_widget = QVTKRenderWindowInteractor(self.key_render_dialog)
        key_render_layout.addWidget(self.key_vtk_widget)

        self.key_renderer = vtk.vtkRenderer()
        self.key_vtk_widget.GetRenderWindow().AddRenderer(self.key_renderer)
        self.key_render_window_interactor = self.key_vtk_widget.GetRenderWindow().GetInteractor()
        self.key_render_window_interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

        self.key_render_dialog.setLayout(key_render_layout)

        self.key_render_dialog.show()
        # 开始交互
        self.key_vtk_widget.Initialize()
        self.key_vtk_widget.Start()

    def show_key_points(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Key Points")
        layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z",
                                         "Angle_X", "Angle_Y", "Angle_Z",
                                         "Physical X", "Physical Y", "Physical Z"])
        table.setRowCount(len(self.key_points))

        for i, (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in enumerate(self.key_points):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            table.setItem(i, 0, name_item)
            table.setItem(i, 1, QTableWidgetItem(f"{x:.2f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{y:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{z:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{angle_x:.2f}"))
            table.setItem(i, 5, QTableWidgetItem(f"{angle_y:.2f}"))
            table.setItem(i, 6, QTableWidgetItem(f"{angle_z:.2f}"))
            table.setItem(i, 8, QTableWidgetItem(f"{phy_x:.2f}"))
            table.setItem(i, 7, QTableWidgetItem(f"{phy_y:.2f}"))
            table.setItem(i, 9, QTableWidgetItem(f"{phy_z:.2f}"))

        table.cellChanged.connect(lambda row, column: self.save_key_points(table, row, column))

        layout.addWidget(table)

        # 添加导出按钮
        export_button = QPushButton("Export to Excel", dialog)
        export_button.clicked.connect(self.export_key_points_to_excel)
        layout.addWidget(export_button)

        dialog.setLayout(layout)
        dialog.show()

    def show_distances(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Distances")
        layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Name", "Distance"])
        table.setRowCount(len(self.distances))

        for i, (name, distance) in enumerate(self.distances):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            table.setItem(i, 0, name_item)
            table.setItem(i, 1, QTableWidgetItem(f"{distance:.2f}"))

        table.cellChanged.connect(lambda row, column: self.save_distances(table, row, column))

        layout.addWidget(table)

        # 添加导出按钮
        export_button = QPushButton("Export to Excel", dialog)
        export_button.clicked.connect(self.export_distances_to_excel)
        layout.addWidget(export_button)

        dialog.setLayout(layout)
        dialog.show()

    def show_angles(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Angles")
        layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Name", "Angle"])
        table.setRowCount(len(self.angles))

        for i, (name, angle) in enumerate(self.angles):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            table.setItem(i, 0, name_item)
            table.setItem(i, 1, QTableWidgetItem(f"{angle:.2f}"))

        table.cellChanged.connect(lambda row, column: self.save_angles(table, row, column))

        layout.addWidget(table)

        # 添加导出按钮
        export_button = QPushButton("Export to Excel", dialog)
        export_button.clicked.connect(self.export_angles_to_excel)
        layout.addWidget(export_button)

        dialog.setLayout(layout)
        dialog.show()

    def convert_row_to_point(self, row):

        pointA = [
            row[0],  # Name 保持为字符串
            round(float(row[1])),
            round(float(row[2])),
            round(float(row[3])),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
            float(row[8]),
            float(row[9])
        ]

        point = self.calculate_position_in_key_coordinates(pointA[1], pointA[2], pointA[3],
                                                           pointA[4], pointA[5], pointA[6])
        return point

    def select_two_points(self, table):
        # 在表格中选择两个点。点击序号会令序号加粗加黑，视为已经选择。
        selected_items = table.selectedItems()
        selected_rows = list(set(item.row() for item in selected_items))

        if len(selected_rows) != 2:
            QMessageBox.warning(None, "Selection Error", "Please select exactly two rows.")
            return

        row1 = [table.item(selected_rows[0], col).text() for col in range(table.columnCount())]
        row2 = [table.item(selected_rows[1], col).text() for col in range(table.columnCount())]


        self.measuring = True

        pointA = self.convert_row_to_point(row1)
        pointB = self.convert_row_to_point(row2)

        self.show_key_render_window()
        self.add_marker_in_3d(self.key_renderer, pointA)
        self.add_marker_in_3d(self.key_renderer, pointB)
        self.draw_line_in_3d(pointA, pointB, True, self.key_renderer)

        self.measuring = False

    def select_three_points(self, table):
        selected_items = table.selectedItems()
        selected_rows = list(set(item.row() for item in selected_items))

        if len(selected_rows) != 3:
            QMessageBox.warning(None, "Selection Error", "Please select exactly two rows.")
            return

        row1 = [table.item(selected_rows[0], col).text() for col in range(table.columnCount())]
        row2 = [table.item(selected_rows[1], col).text() for col in range(table.columnCount())]
        row3 = [table.item(selected_rows[2], col).text() for col in range(table.columnCount())]

        point1 = self.convert_row_to_point(row1)
        point2 = self.convert_row_to_point(row2)
        point3 = self.convert_row_to_point(row3)

        self.measuring = False
        self.measuring_angle = True
        self.show_key_render_window()
        self.add_marker_in_3d(self.key_renderer, point1)
        self.add_marker_in_3d(self.key_renderer, point2)
        self.add_marker_in_3d(self.key_renderer, point3)

        self.draw_line_in_3d(point1, point2, True, self.key_renderer)
        self.draw_line_in_3d(point2, point3, True, self.key_renderer)
        self.label_angle_in_3d(point1, point2, point3, self.key_renderer)

        self.measuring_angle = False

    def select_horizontal_angle(self, table):
        selected_items = table.selectedItems()
        selected_rows = list(set(item.row() for item in selected_items))

        if len(selected_rows) != 2:
            QMessageBox.warning(None, "Selection Error", "Please select exactly two rows.")
            return

        row1 = [table.item(selected_rows[0], col).text() for col in range(table.columnCount())]
        row2 = [table.item(selected_rows[1], col).text() for col in range(table.columnCount())]


        self.measuring = False

        pointA = self.convert_row_to_point(row1)
        pointB = self.convert_row_to_point(row2)

        self.show_key_render_window()
        self.add_marker_in_3d(self.key_renderer, pointA)
        self.add_marker_in_3d(self.key_renderer, pointB)
        self.draw_line_in_3d(pointA, pointB, True, self.key_renderer)

        self.measuring_angle = True

        pointC = [0, 0, pointB[2]]
        # self.show_key_render_window()
        self.label_angle_in_3d(pointA, pointB, pointC, self.key_renderer)
        self.add_plane_xy(pointC, self.key_renderer)

        self.measuring_angle = False

    def draw_line_in_3d(self, point1, point2, drawing, renderer):
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(point1)
        line_source.SetPoint2(point2)

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_source.GetOutputPort())

        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0, 1, 0)  # 绿色

        if drawing:
            renderer.AddActor(line_actor)

        # 计算距离
        if self.measuring and not self.measuring_angle and not self.measuring_horizontal_angle:
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            distance *= self.slice_thickness
            self.display_distance_in_3d(renderer, point1, point2, distance)
            self.last_distance_label.setText(f"Last Distance: {distance:.2f} mm")

        renderer.ResetCamera()
        renderer.GetRenderWindow().Render()

    def display_distance_in_3d(self, renderer, point1, point2, distance):
        text_source = vtk.vtkTextSource()
        text_source.SetText(f"{distance:.2f} mm")
        text_source.SetBackgroundColor(1.0, 1.0, 1.0)
        text_source.SetForegroundColor(1.0, 0.0, 0.0)
        text_source.Update()

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        text_actor = vtk.vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetScale(0.5, 0.5, 0.5)
        mid_point = (np.array(point1) + np.array(point2)) / 2

        text_actor.SetPosition(mid_point[0] + 1, mid_point[1] + 1, mid_point[2] + 1)
        text_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        text_actor.SetCamera(renderer.GetActiveCamera())

        renderer.AddActor(text_actor)
        renderer.Render()

    def add_marker_in_3d(self, renderer, world_pos):
        point_source = vtk.vtkPointSource()
        point_source.SetCenter(world_pos)
        point_source.SetNumberOfPoints(1)

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(self.marker_radius)

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetInputConnection(point_source.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # 红色

        renderer.AddActor(actor)
        renderer.Render()

        '''
        actor_center = actor.GetCenter()

        return actor_center 
        #红点的实际中心与点击位置有非常细微的差距，推测是由于浮点计算的精度问题导致，目前未找到解决办法
        #所以最终选择将红点实际中心返还并加入表格中，以确保图像中红点中心与表格对齐，特此注明
        '''

    def add_plane_xy(self, point, renderer):
        # 创建平面源
        normal = [0, 0, 1]
        plane_source = vtk.vtkPlaneSource()

        # 设置平面的中心和法向量
        plane_source.SetOrigin(0, 0, 0)
        plane_source.SetPoint1(768, 0, 0)
        plane_source.SetPoint2(0, 768, 0)

        plane_source.SetCenter(point)
        plane_source.SetNormal(normal)

        # 设置平面的尺寸
        plane_source.SetXResolution(50)
        plane_source.SetYResolution(50)

        # 更新平面源
        plane_source.Update()

        # 映射平面数据
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane_source.GetOutputPort())

        # 创建平面演员
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)  # 设置平面颜色，绿色

        renderer.AddActor(actor)
        return actor

    def label_angle_in_3d(self, point1, point2, point3, renderer):
        # 计算角度
        angle = self.calculate_angle(point1, point2, point3)

        text_source = vtk.vtkTextSource()

        text_source.SetText(f"{angle:.2f}°")
        text_source.SetBackgroundColor(1.0, 1.0, 1.0)
        text_source.SetForegroundColor(1.0, 0.0, 0.0)
        text_source.Update()

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        text_actor = vtk.vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetScale(0.5, 0.5, 0.5)

        text_actor.SetPosition(point2[0], point2[1] + 1, point2[2] + 1)
        text_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        text_actor.SetCamera(renderer.GetActiveCamera())

        renderer.AddActor(text_actor)
        renderer.Render()

        if self.measuring_angle:
            self.last_angle_label.setText(f"Last Angle: {angle:.2f} °")
        elif self.measuring_horizontal_angle:
            self.last_h_angle_label.setText(f"Last Horizontal Angle: {angle:.2f} °")

    def export_to_excel(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save as Excel file", "", "Excel Files (*.xlsx)")
        if file_path:
            data = []
            for name, point, angle, physical in self.marked_points:
                data.append([name, round(point[0], 2), round(point[1], 2), round(point[2], 2),
                             angle[0], angle[1], angle[2],
                             round(physical[1], 2), round(physical[0], 2), round(physical[2], 2)])
            print(data[-1])
            df = pd.DataFrame(data, columns=["Name", "X", "Y", "Z", "Angle X", "Angle Y", "Angle Z", "Physical X",
                                             "Physical Y", "Physical Z"])
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Export Successful", f"Points have been exported to {file_path}")

    def export_key_points_to_excel(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save as Excel file", "", "Excel Files (*.xlsx)")
        if file_path:
            data = []
            for name, point, angle, physical in self.key_points:
                data.append([name, round(point[0], 2), round(point[1], 2), round(point[2], 2),
                             round(self.euler_angles[0], 2), round(self.euler_angles[1], 2),
                             round(self.euler_angles[2], 2),
                             round(physical[1], 2), round(physical[0], 2), round(physical[2], 2)])
            df = pd.DataFrame(data, columns=["Name", "X", "Y", "Z", "Angle X", "Angle Y", "Angle Z", "Physical X",
                                             "Physical Y", "Physical Z"])
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Export Successful", f"Points have been exported to {file_path}")

    def export_angles_to_excel(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save as Excel file", "", "Excel Files (*.xlsx)")
        if file_path:
            data = []
            for name, angle in self.angles:
                data.append([name, angle])
            df = pd.DataFrame(data, columns=["Name", "Angle"])
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Export Successful", f"Points have been exported to {file_path}")

    def export_distances_to_excel(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save as Excel file", "", "Excel Files (*.xlsx)")
        if file_path:
            data = []
            for name, distance in self.distances:
                data.append([name, distance])
            df = pd.DataFrame(data, columns=["Name", "distance"])
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "Export Successful", f"Points have been exported to {file_path}")

    def show_minimenu(self, obj, event):
        interactor = obj.GetRenderWindow().GetInteractor()
        if not interactor.GetControlKey():  # 检查是否按下了 Ctrl 键，如按下就不显示小菜单
            context_menu = QMenu(self)

            mark_action = context_menu.addAction("标记现在点")
            mark_action.triggered.connect(self.enable_marking)

            measure_simple_action = context_menu.addAction("测量两点距离")
            measure_simple_action.triggered.connect(self.measure_two_points)

            measure_angle_simple_action = context_menu.addAction("测量三点角度")
            measure_angle_simple_action.triggered.connect(self.measure_one_angle)

            measure_horizontal_angle_action = context_menu.addAction("测量水平角度")
            measure_horizontal_angle_action.triggered.connect(self.measure_angle_horizontal)

            erase_action = context_menu.addAction("橡皮擦开")
            erase_action.triggered.connect(self.enable_erasing)

            disable_erase_action = context_menu.addAction("橡皮擦关")
            disable_erase_action.triggered.connect(self.disable_erasing)

            context_menu.exec(QCursor.pos())

    def show_slice_position_in_3d(self):
        self.projection_3d = True

        # 获取当前切片序列号
        x = self.x_input.value()
        y = self.y_input.value()
        z = self.z_input.value()

        pos = self.rotate_coordinate(x, y, z,
                                     self.euler_angles[0],
                                     self.euler_angles[1],
                                     self.euler_angles[2], True)
        x = pos[0]
        y = pos[1]
        z = pos[2]

        # 清除旧的标记和线条
        if self.slice_marker:
            self.renderer_3d.RemoveActor(self.slice_marker)
        if self.x_line_actor:
            self.renderer_3d.RemoveActor(self.x_line_actor)
        if self.y_line_actor:
            self.renderer_3d.RemoveActor(self.y_line_actor)
        if self.z_line_actor:
            self.renderer_3d.RemoveActor(self.z_line_actor)

        # 创建红点
        point_source = vtk.vtkSphereSource()
        point_source.SetCenter(- x, y, z)
        point_source.SetRadius(5.0)

        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputConnection(point_source.GetOutputPort())

        self.slice_marker = vtk.vtkActor()
        self.slice_marker.SetMapper(point_mapper)
        self.slice_marker.GetProperty().SetColor(1, 0, 0)

        self.renderer_3d.AddActor(self.slice_marker)

        # 创建X轴平行线
        line_source_x = vtk.vtkLineSource()
        line_source_x.SetPoint1(0, y, z)
        line_source_x.SetPoint2(-self.width + 1, y, z)

        line_mapper_x = vtk.vtkPolyDataMapper()
        line_mapper_x.SetInputConnection(line_source_x.GetOutputPort())

        self.x_line_actor = vtk.vtkActor()
        self.x_line_actor.SetMapper(line_mapper_x)
        self.x_line_actor.GetProperty().SetColor(0, 0, 1)

        self.renderer_3d.AddActor(self.x_line_actor)

        # 创建Y轴平行线
        line_source_y = vtk.vtkLineSource()
        line_source_y.SetPoint1(- x, 0, z)
        line_source_y.SetPoint2(- x, self.height, z)

        line_mapper_y = vtk.vtkPolyDataMapper()
        line_mapper_y.SetInputConnection(line_source_y.GetOutputPort())

        self.y_line_actor = vtk.vtkActor()
        self.y_line_actor.SetMapper(line_mapper_y)
        self.y_line_actor.GetProperty().SetColor(0, 1, 0)

        self.renderer_3d.AddActor(self.y_line_actor)

        # 创建Z轴平行线
        line_source_z = vtk.vtkLineSource()
        line_source_z.SetPoint1(-x, y, 0)
        line_source_z.SetPoint2(-x, y, self.depth)

        line_mapper_z = vtk.vtkPolyDataMapper()
        line_mapper_z.SetInputConnection(line_source_z.GetOutputPort())

        self.z_line_actor = vtk.vtkActor()
        self.z_line_actor.SetMapper(line_mapper_z)
        self.z_line_actor.GetProperty().SetColor(1, 1, 0)

        self.renderer_3d.AddActor(self.z_line_actor)
        self.render_window_3d.Render()

    def stop_showing_3d(self):
        self.projection_3d = False
        # 清除旧的标记和线条
        if self.slice_marker:
            self.renderer_3d.RemoveActor(self.slice_marker)
        if self.x_line_actor:
            self.renderer_3d.RemoveActor(self.x_line_actor)
        if self.y_line_actor:
            self.renderer_3d.RemoveActor(self.y_line_actor)
        if self.z_line_actor:
            self.renderer_3d.RemoveActor(self.z_line_actor)

        self.render_window_3d.Render()

    def save_marked_points(self, table, row, column):
        if column == 0:  # Only update if the name column is changed
            name_item = table.item(row, 0).text()
            x_item = float(table.item(row, 1).text())
            y_item = float(table.item(row, 2).text())
            z_item = float(table.item(row, 3).text())
            angle_x_item = float(table.item(row, 4).text())
            angle_y_item = float(table.item(row, 5).text())
            angle_z_item = float(table.item(row, 6).text())
            phy_x_item = float(table.item(row, 7).text())
            phy_y_item = float(table.item(row, 8).text())
            phy_z_item = float(table.item(row, 9).text())
            self.marked_points[row] = (name_item, (x_item, y_item, z_item),
                                       (angle_x_item, angle_y_item, angle_z_item),
                                       (phy_x_item, phy_y_item, phy_z_item))

    def save_key_points(self, table, row, column):
        if column == 0:  # Only update if the name column is changed
            name_item = table.item(row, 0).text()
            x_item = float(table.item(row, 1).text())
            y_item = float(table.item(row, 2).text())
            z_item = float(table.item(row, 3).text())
            angle_x_item = float(table.item(row, 4).text())
            angle_y_item = float(table.item(row, 5).text())
            angle_z_item = float(table.item(row, 6).text())
            phy_x_item = float(table.item(row, 7).text())
            phy_y_item = float(table.item(row, 8).text())
            phy_z_item = float(table.item(row, 9).text())
            self.key_points[row] = (name_item, (x_item, y_item, z_item),
                                    (angle_x_item, angle_y_item, angle_z_item),
                                    (phy_x_item, phy_y_item, phy_z_item))

    def save_distances(self, table, row, column):
        if column == 0:  # Only update if the name column is changed
            name_item = table.item(row, 0).text()
            distance_item = float(table.item(row, 1).text())
            self.distances[row] = (name_item, distance_item)

    def save_angles(self, table, row, column):
        if column == 0:  # Only update if the name column is changed
            name_item = table.item(row, 0).text()
            angle_item = float(table.item(row, 1).text())
            self.angles[row] = (name_item, angle_item)

    def save_screenshot(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            screenshot = self.grab()
            screenshot.save(file_path)
            # Save the screenshots of the axial, coronal, and sagittal windows
            self.clear_lines()
            self.save_vtk_render_window(self.vtk_widget_axial, file_path.replace('.png', '_axial_labeled.png'))
            self.save_vtk_render_window(self.vtk_widget_coronal, file_path.replace('.png', '_coronal_labeled.png'))
            self.save_vtk_render_window(self.vtk_widget_sagittal, file_path.replace('.png', '_sagittal_labeled.png'))
            self.save_vtk_render_window(self.vtk_widget_3d, file_path.replace('.png', '_3d.png'))
            self.clear_marker_and_line()
            self.save_vtk_render_window(self.vtk_widget_axial, file_path.replace('.png', '_axial_clean.png'))
            self.save_vtk_render_window(self.vtk_widget_coronal, file_path.replace('.png', '_coronal_clean.png'))
            self.save_vtk_render_window(self.vtk_widget_sagittal, file_path.replace('.png', '_sagittal_clean.png'))
        self.update_views()

    def save_vtk_render_window(self, vtk_widget, file_path):
        render_window = vtk_widget.GetRenderWindow()
        renderer = render_window.GetRenderers().GetFirstRenderer()

        # Reset the camera to capture the entire slice
        renderer.ResetCamera()
        # render_window.Render()

        # Capture the image from the render window
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.SetScale(5)  # Adjust the scale for higher resolution
        window_to_image_filter.Update()

        # Save the captured image
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(file_path)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

        # Clean up
        render_window.Finalize()

    def save_axial_window(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            self.clear_lines()
            self.save_vtk_render_window(self.vtk_widget_axial, file_path.replace('.png', '_axial_labeled.png'))
            self.clear_marker_and_line()
            self.save_vtk_render_window(self.vtk_widget_axial, file_path.replace('.png', '_axial_clean.png'))

    def save_coronal_window(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            self.clear_lines()
            self.save_vtk_render_window(self.vtk_widget_coronal, file_path.replace('.png', '_coronal_labeled.png'))
            self.clear_marker_and_line()
            self.save_vtk_render_window(self.vtk_widget_coronal, file_path.replace('.png', '_coronal_clean.png'))

    def save_sagittal_window(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            self.clear_lines()
            self.save_vtk_render_window(self.vtk_widget_sagittal, file_path.replace('.png', '_sagittal_labeled.png'))
            self.clear_marker_and_line()
            self.save_vtk_render_window(self.vtk_widget_sagittal, file_path.replace('.png', '_sagittal_clean.png'))

    def save_3d_window(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            temp = self.projection_3d
            if not temp:
                self.show_slice_position_3d()
            self.renderer_3d.RemoveActor(self.x_line_actor)  # x轴
            self.renderer_3d.RemoveActor(self.y_line_actor)  # y轴
            self.renderer_3d.RemoveActor(self.z_line_actor)  # z轴
            self.save_vtk_render_window(self.vtk_widget_3d, file_path.replace('.png', '_3d_labeled.png'))
            self.renderer_3d.RemoveActor(self.slice_marker)
            self.save_vtk_render_window(self.vtk_widget_3d, file_path.replace('.png', '_3d_clean.png'))
            if temp:
                self.show_slice_position_3d()

    def save_state_snapshot(self):
        # 获取当前XYZ坐标和旋转角度
        x = self.x_input.value()
        y = self.y_input.value()
        z = self.z_input.value()
        rotate_x = self.rotate_x_input.value()
        rotate_y = self.rotate_y_input.value()
        rotate_z = self.rotate_z_input.value()

        snapshot = StateSnapshot(x, y, z, rotate_x, rotate_y, rotate_z)
        self.state_snapshots.append(snapshot)

    def restore_state_snapshot(self):
        if self.state_snapshots:
            snapshot = self.state_snapshots.pop()
            self.x_input.setValue(snapshot.x)
            self.y_input.setValue(snapshot.y)
            self.z_input.setValue(snapshot.z)
            self.rotate_x_input.setValue(snapshot.rotate_x)
            self.rotate_y_input.setValue(snapshot.rotate_y)
            self.rotate_z_input.setValue(snapshot.rotate_z)
            self.update_views()  # 更新视图以反映恢复的状态

    def undo(self):
        self.restore_state_snapshot()

    def param_init(self,dicom_viewer):
        self.slice_thickness = dicom_viewer.slice_thickness
        self.marked_points = dicom_viewer.marked_points
        self.key_points = dicom_viewer.key_points
        self.distances = dicom_viewer.distances
        self.angles = dicom_viewer.angles
        self.markers = dicom_viewer.markers  # 储存现在坐标在三视图上的红点actor
        self.lines = dicom_viewer.lines  # 储存线条actor
        self.state_snapshots = dicom_viewer.state_snapshots  # 保存最近的三次状态快照
        self.system = dicom_viewer.system

        self.AODA = dicom_viewer.AODA
        self.ANS = dicom_viewer.ANS
        self.HtR = dicom_viewer.HtR
        self.HtL = dicom_viewer.HtL
        self.SR = dicom_viewer.SR

        self.last_distance_label.setText(f"Last Distance: {0.00} mm")
        self.last_angle_label.setText(f"Last Angle: {0.00} °")
        self.last_h_angle_label.setText(f"Last Horizontal Angle: {0.00} °")

        self.lr_count = dicom_viewer.lr_count
        self.fh_count = dicom_viewer.fh_count
        self.tb_count = dicom_viewer.tb_count


    def open_files(self):
        file_dialog = QFileDialog(self, "Select DICOM File(s)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters(["DICOM Files (*.dcm)", "All Files (*)"])

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                dcm = DICOMViewer(selected_files)
                self.dicom_viewers.append(dcm)  # 将实例添加到列表中

            # 更新当前选择的DICOMViewer
            self.current_viewer_index = len(self.dicom_viewers) - 1 # 默认新打开的
            self.param_init(self.dicom_viewers[self.current_viewer_index])
            self.visualize_vtk_image(self.dicom_viewers[self.current_viewer_index].vtk_image)

    def switch_image(self):
        if len(self.dicom_viewers) > 1:
            self.current_viewer_index = (self.current_viewer_index + 1) % len(self.dicom_viewers)
            self.param_init(self.dicom_viewers[self.current_viewer_index])
            self.visualize_vtk_image(self.dicom_viewers[self.current_viewer_index].vtk_image)

            for _ in range(self.lr_count):
                self.flip_LR_AUTO()
            for _ in range(self.fh_count):
                self.flip_FH_AUTO()
            for _ in range(self.tb_count):
                self.flip_TB_AUTO()

            if self.system == 1:
                self.set_coordinate_system()

    def compare_window(self, view_type):
        """
        此方法创建一个新窗口并根据选择的视图类型同时显示两个 DICOM 文件的切片图，并设置第一个图像的透明度为50%。

        :param view_type: 选择视图类型，'axial', 'coronal' 或 'sagittal'
        """
        # 获取当前选中的 DICOMViewer 实例
        dicom_viewer_1 = self.dicom_viewers[0]  # 第一个 DICOMViewer 实例
        dicom_viewer_2 = self.dicom_viewers[1]  # 第二个 DICOMViewer 实例

        # 确保两个 DICOM 图像已加载
        if dicom_viewer_1.vtk_image is None or dicom_viewer_2.vtk_image is None:
            print("Error: One or both DICOM images are not loaded.")
            return
        print("DICOM images loaded successfully.")

        # 打印 DICOM 图像的维度
        print(f"Depth: {dicom_viewer_1.depth}, Height: {dicom_viewer_1.height}, Width: {dicom_viewer_1.width}")
        print(f"Depth: {dicom_viewer_2.depth}, Height: {dicom_viewer_2.height}, Width: {dicom_viewer_2.width}")

        # 创建并初始化新界面
        new_window = QMainWindow(self)
        new_window.setWindowTitle("Compare DICOM Files")

        # 进行初始化操作
        central_widget = QWidget(new_window)
        new_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建视图窗口
        vtk_widget = QVTKRenderWindowInteractor(central_widget)
        layout.addWidget(vtk_widget)

        # 创建渲染器
        renderer = vtk.vtkRenderer()
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        interactor = vtk_widget.GetRenderWindow().GetInteractor()

        # 创建第一个 ResliceImageViewer（用于显示第一个 DICOM 图像）
        def create_image_actor(vtk_image, slice_orientation):
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(vtk_image)
            reslice.SetOutputDimensionality(2)

            if slice_orientation == 'axial':
                reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 1, 0, 0, 0, 1)
                reslice.SetResliceAxesOrigin(0, 0, vtk_image.GetDimensions()[2] // 2)
            elif slice_orientation == 'coronal':
                reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, 1, 0)
                reslice.SetResliceAxesOrigin(0, 768 - 316, 0)
            elif slice_orientation == 'sagittal':
                reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)
                reslice.SetResliceAxesOrigin(vtk_image.GetDimensions()[0] // 2, 0, 0)

            reslice.SetInterpolationModeToLinear()

            actor = vtk.vtkImageActor()
            actor.GetMapper().SetInputConnection(reslice.GetOutputPort())
            return actor

        # 创建图像actor
        image_actor1 = create_image_actor(dicom_viewer_1.vtk_image, view_type)
        image_actor2 = create_image_actor(dicom_viewer_2.vtk_image, view_type)

        # 设置第一个图像为半透明，第二个为不透明
        image_actor1.GetProperty().SetOpacity(0.5)
        image_actor2.GetProperty().SetOpacity(1.0)

        # 设置图像的颜色窗口（亮度/对比度）
        image_actor1.GetProperty().SetColorWindow(600)
        image_actor2.GetProperty().SetColorWindow(600)

        # 将图像actor添加到渲染器
        renderer.AddActor(image_actor1)
        renderer.AddActor(image_actor2)

        # 初始化 MouseInteractorStyle，并设置 image_actor2 为可拖动的对象
        self.style = MouseInteractorStyle(renderer, image_actor2)
        interactor.SetInteractorStyle(self.style)

        # 渲染图像
        renderer.ResetCamera()
        vtk_widget.GetRenderWindow().Render()
        interactor.Initialize()

        # 创建切换图像透明度的按钮
        change_button = QPushButton('Change Image', new_window)
        change_button.clicked.connect(lambda: self.change_image(image_actor1, image_actor2))

        # 将按钮添加到布局
        layout.addWidget(change_button)

        # 设置新窗口的固定大小
        new_window.setFixedSize(800, 600)  # 设置宽度800px，高度600px，并固定大小

        # 显示新窗口
        new_window.show()

    def change_image(self, image_actor1, image_actor2):
        """
        切换图像的透明度，当前显示的第一个图像为半透明，第二个图像为不透明，点击按钮时切换状态
        """
        # 切换透明度
        current_opacity_1 = image_actor1.GetProperty().GetOpacity()
        current_opacity_2 = image_actor2.GetProperty().GetOpacity()

        # 如果第一个图像是透明的，则设置第一个图像为不透明，第二个图像为半透明
        if current_opacity_1 == 0.5:
            image_actor1.GetProperty().SetOpacity(1.0)
            image_actor2.GetProperty().SetOpacity(0.5)
        else:
            # 否则，设置第一个图像为半透明，第二个图像为不透明
            image_actor1.GetProperty().SetOpacity(0.5)
            image_actor2.GetProperty().SetOpacity(1.0)

        # 渲染图像
        image_actor1.GetRenderer().GetRenderWindow().Render()
        image_actor2.GetRenderer().GetRenderWindow().Render()

    def add_right_click_zoom_handler(self, interactor, viewer):
        """Add a custom right-click listener for zoom functionality."""
        """Remove existing right-click observers from the interactor."""
        events_to_remove = ["RightButtonPressEvent", "RightButtonReleaseEvent"]
        for event in events_to_remove:
            interactor.RemoveObservers(event)

        scaling_factor = 0.005  # 缩放速率

        self.zooming = False
        self.last_y_position = 0

        def on_right_button_press_zoom(obj, event):
            if interactor.GetControlKey():  # 检查是否按下了 Ctrl 键
                self.zooming = True
                self.last_y_position = interactor.GetEventPosition()[1]
                # Start custom zoom mode
                print("Zoom mode activated (right-click).")

        def on_mouse_move_zoom(obj, event):
            if self.zooming:
                current_y_position = interactor.GetEventPosition()[1]
                delta_y = current_y_position - self.last_y_position
                self.last_y_position = current_y_position

                # 获取当前缩放比例
                current_zoom = viewer.GetRenderer().GetActiveCamera().GetParallelScale()
                # print(f"Before Zoom: {current_zoom}, Delta Y: {delta_y}")

                # 根据拖动方向调整缩放
                zoom_factor = 1 - scaling_factor * abs(delta_y)
                if delta_y > 0:  # 向下拖动缩小
                    new_zoom = np.abs(current_zoom * zoom_factor)
                else:  # 向上拖动放大
                    new_zoom = np.abs(current_zoom / zoom_factor)

                # 限制缩放范围
                # new_zoom = max(min_zoom, min(new_zoom, max_zoom))
                # print(f"After Zoom: {new_zoom}")

                # 更新缩放值
                viewer.GetRenderer().GetActiveCamera().SetParallelScale(new_zoom)
                viewer.Render()

        def on_right_button_release_zoom(obj, event):
            self.zooming = False
            print("Zoom mode deactivated (right-click).")

        interactor.AddObserver("RightButtonPressEvent", on_right_button_press_zoom)
        interactor.AddObserver("MouseMoveEvent", on_mouse_move_zoom)
        interactor.AddObserver("RightButtonReleaseEvent", on_right_button_release_zoom)
        # interactor.AddObserver("LeftButtonReleaseEvent", on_right_button_release_zoom)

    def flip_vtk_image(self, vtk_image, axis):
        flip = vtk.vtkImageFlip()
        flip.SetInputData(vtk_image)
        flip.SetFilteredAxis(axis)  # 0 = x, 1 = y, 2 = z
        flip.Update()
        return flip.GetOutput()
        '''
        def adjust_orientation(self, vtk_image, image_orientation_patient):
            iop = self.image_orientation_patient
            patient_position = self.patient_position
            print(iop)
            print(patient_position)
            1. 根据ImageOrientationPatient调整方向
            IOP的六个数值分别表示方向向量：
            iop[0:3] 表示图像行的方向向量，iop[3:6] 表示图像列的方向向量
            我们根据这些向量来判断图像需要如何翻转和旋转

            2. Axial切片调整
            if iop == [1, 0, 0, 0, 1, 0]:  # 如果图像已经是Axial方向
                if patient_position in ['HFS', 'FFS']:  # Head First Supine 或 Feet First Supine
                    vtk_image = self.flip_vtk_image(vtk_image, 1)  # 翻转Y轴
                elif patient_position in ['HFP', 'FFP']:  # Head First Prone 或 Feet First Prone
                    vtk_image = self.flip_vtk_image(vtk_image, 2)  # 翻转Z轴

            3. Coronal切片调整
            elif iop == [0, 1, 0, 0, 0, 1]:  # 如果图像是Coronal方向
                if patient_position in ['HFS', 'HFP']:  # Head First Supine 或 Prone
                    vtk_image = self.flip_vtk_image(vtk_image, 0)  # 翻转X轴
                elif patient_position in ['FFS', 'FFP']:  # Feet First Supine 或 Prone
                    vtk_image = self.flip_vtk_image(vtk_image, 2)  # 翻转Z轴

            4. Sagittal切片调整
            elif iop == [0, 0, 1, 0, 1, 0]:  # 如果图像是Sagittal方向
                if patient_position in ['HFS', 'HFP']:  # Head First Supine 或 Prone
                    vtk_image = self.flip_vtk_image(vtk_image, 0)  # 翻转X轴
                elif patient_position in ['FFS', 'FFP']:  # Feet First Supine 或 Prone
                    vtk_image = self.flip_vtk_image(vtk_image, 2)  # 翻转Z轴

            return vtk_image
            '''

    def flip_LR(self):  # 左右镜像
        self.flipped_image = self.flip_vtk_image(self.flipped_image, 0)
        self.flip = True
        self.visualize_vtk_image(self.flipped_image)
        self.flip = False

        self.dicom_viewers[self.current_viewer_index].lr_count = self.dicom_viewers[self.current_viewer_index].lr_count + 1

    def flip_LR_AUTO(self):  # 左右镜像
        self.flipped_image = self.flip_vtk_image(self.flipped_image, 0)
        self.flip = True
        self.visualize_vtk_image(self.flipped_image)
        self.flip = False

    def flip_FH(self):  # 前后镜像
        self.flipped_image = self.flip_vtk_image(self.flipped_image, 2)
        self.flip = True
        self.visualize_vtk_image(self.flipped_image)
        self.flip = False

        self.dicom_viewers[self.current_viewer_index].fh_count = self.dicom_viewers[self.current_viewer_index].fh_count + 1

    def flip_FH_AUTO(self):  # 前后镜像
        self.flipped_image = self.flip_vtk_image(self.flipped_image, 2)
        self.flip = True
        self.visualize_vtk_image(self.flipped_image)
        self.flip = False

    def flip_TB(self):  # 上下镜像
        self.flipped_image = self.flip_vtk_image(self.flipped_image, 1)
        self.flip = True
        self.visualize_vtk_image(self.flipped_image)
        self.flip = False

        self.dicom_viewers[self.current_viewer_index].tb_count = self.dicom_viewers[self.current_viewer_index].tb_count + 1

    def flip_TB_AUTO(self):  # 上下镜像
        self.flipped_image = self.flip_vtk_image(self.flipped_image, 1)
        self.flip = True
        self.visualize_vtk_image(self.flipped_image)
        self.flip = False

    def rotate_vtk_image(self, vtk_image, angle, axis):
        transform = vtk.vtkTransform()

        # 获取图像的中心
        center = [0] * 3
        vtk_image.GetCenter(center)

        # 先将图像中心移到原点
        transform.Translate(center[0], center[1], center[2])

        # 旋转图像
        transform.RotateWXYZ(angle, [1 if i == axis else 0 for i in range(3)])

        # 再将图像移回原位置
        transform.Translate(-center[0], -center[1], -center[2])

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(vtk_image)
        reslice.SetResliceTransform(transform)
        reslice.SetInterpolationModeToLinear()
        reslice.Update()

        return reslice.GetOutput()

    def setup_camera(self, renderer):
        camera = vtk.vtkCamera()
        # 设置相机位置，使得X轴方向与二维视图一致
        camera.SetPosition(-1, 0, 0)
        camera.SetViewUp(0, 0, 1)
        camera.SetFocalPoint(0, 0, 0)
        renderer.SetActiveCamera(camera)
        renderer.ResetCamera()

    def visualize_vtk_image(self, vtk_image):

        # 先清除掉以前渲染器中的所有演员
        if not self.first_open:
            renderer_axial = self.axial_viewer.GetRenderer()
            axial_removes = self.pick_actors_in_radius(renderer_axial, [0, 0, 0], 2000)
            if axial_removes:
                for picked_actor in axial_removes:
                    renderer_axial.RemoveActor(picked_actor)

            renderer_coronal = self.coronal_viewer.GetRenderer()
            coronal_removes = self.pick_actors_in_radius(renderer_coronal, [0, 0, 0], 2000)
            if coronal_removes:
                for picked_actor in coronal_removes:
                    renderer_coronal.RemoveActor(picked_actor)

            renderer_sagittal = self.sagittal_viewer.GetRenderer()
            sagittal_removes = self.pick_actors_in_radius(renderer_sagittal, [0, 0, 0], 2000)
            if sagittal_removes:
                for picked_actor in sagittal_removes:
                    renderer_sagittal.RemoveActor(picked_actor)

            # 初始化变量，控制监听器是否监听任务。
            self.picking = False  # 开启三视图点击联动
            self.measuring = False  # 测距离
            self.measuring_angle = False  # 测角度
            self.measuring_horizontal_angle = False  # 测水平角度
            self.marking = False  # 是否标记
            self.erasing = False  # 擦除
            self.projection_3d = False  # 是否进行3D映射
            self.projection_3d_2d = False  # 是否开启3D反射回2D
            self.coords_plane_display = False  # 是否展示坐标平面


            self.erase_radius = 20.0  # 初始化擦除半径

            self.marker_radius = 3.0  # 初始化标记红点大小（半径）

            # 3D映射
            self.slice_marker = None  # 现在所处点的marker
            self.x_line_actor = None  # x轴
            self.y_line_actor = None  # y轴
            self.z_line_actor = None  # z轴

            self.euler_angles = [0, 0, 0]

            self.set_SR_button.setStyleSheet("color: black;")
            self.set_AODA_button.setStyleSheet("color: black;")
            self.set_ANS_button.setStyleSheet("color: black;")
            self.set_HtR_button.setStyleSheet("color: black;")
            self.set_HtL_button.setStyleSheet("color: black;")
            self.set_origin_button.setStyleSheet("color: black;")

            self.flip = False

            self.renderer_3d.RemoveAllViewProps()

            self.rotate_x_input.setValue(0)
            self.rotate_y_input.setValue(0)
            self.rotate_z_input.setValue(0)


        if not self.flip:
            self.flipped_image = self.flip_vtk_image(vtk_image, 1)
            self.flipped_image = self.flip_vtk_image(self.flipped_image, 2)
        else:
            self.flipped_image = vtk_image


        dimensions = self.flipped_image.GetDimensions()
        self.width, self.height, self.depth = dimensions

        # 更新输入字段的范围
        self.z_input.setRange(0, self.depth - 1)  # Axial
        self.y_input.setRange(0, self.height - 1)  # Coronal
        self.x_input.setRange(0, self.width - 1)  # Sagittal

        middle_axial = self.depth // 2
        middle_coronal = self.height // 2
        middle_sagittal = self.width // 2

        self.x_input.setValue(middle_sagittal)
        self.y_input.setValue(middle_coronal)
        self.z_input.setValue(middle_axial)

        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetInputData(self.flipped_image)

        # 取图像中心坐标
        self.center = [0] * 3
        self.flipped_image.GetCenter(self.center)

        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetOutputSpacing(1, 1, 1)

        # 初始化分别对应X，Y，Z轴的transform对象用于旋转
        self.transform_x = vtk.vtkTransform()
        self.transform_y = vtk.vtkTransform()
        self.transform_z = vtk.vtkTransform()

        # 设置输出范围
        self.reslice.SetOutputExtent(0, self.width - 1, 0, self.height - 1, 0, self.depth - 1)

        # 使用 vtkResliceImageViewer 显示切片
        self.axial_viewer.SetInputConnection(self.reslice.GetOutputPort())
        self.axial_viewer.SetSliceOrientationToXY()
        self.axial_viewer.SetSlice(middle_axial)
        self.axial_viewer.SetColorWindow(2000)  # 设置初始窗宽（对比度）
        self.axial_viewer.SetColorLevel(-300)  # 设置初始窗位（亮度）
        self.axial_viewer.Render()

        self.coronal_viewer.SetInputConnection(self.reslice.GetOutputPort())
        self.coronal_viewer.SetSliceOrientationToXZ()
        self.coronal_viewer.SetSlice(middle_coronal)
        self.coronal_viewer.SetColorWindow(2000)  # 设置初始窗宽（对比度）
        self.coronal_viewer.SetColorLevel(-300)  # 设置初始窗位（亮度）
        self.coronal_viewer.Render()

        self.sagittal_viewer.SetInputConnection(self.reslice.GetOutputPort())
        self.sagittal_viewer.SetSliceOrientationToYZ()
        self.sagittal_viewer.SetSlice(middle_sagittal)
        self.sagittal_viewer.SetColorWindow(2000)  # 设置初始窗宽（对比度）
        self.sagittal_viewer.SetColorLevel(-300)  # 设置初始窗位（亮度）
        self.sagittal_viewer.Render()

        # 3D 渲染部分

        # 创建一个 transform 对象来反置坐标轴方向
        transform = vtk.vtkTransform()
        # 将 X 轴反置
        transform.Scale(-1, 1, 1)

        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(self.flipped_image)

        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeOff()
        volume_property.SetInterpolationTypeToLinear()

        color_function = vtk.vtkColorTransferFunction()
        range_slice = 2000 - 150
        color_function.AddRGBPoint(150, 0.0, 0.0, 0.0)
        color_function.AddRGBPoint(150 + range_slice * 1 / 4, 1.0, 0.5, 0.3)
        color_function.AddRGBPoint(150 + range_slice * 2 / 4, 1.0, 0.5, 0.3)
        color_function.AddRGBPoint(150 + range_slice * 3 / 4, 1.0, 1.0, 0.9)
        color_function.AddRGBPoint(2000, 1.0, 1.0, 1.0)

        opacity_function = vtk.vtkPiecewiseFunction()
        opacity_function.AddPoint(150, 0.00)
        opacity_function.AddPoint(2000, 1.00)
        volume_property.SetColor(color_function)
        volume_property.SetScalarOpacity(opacity_function)

        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        volume.SetUserTransform(transform)

        self.renderer_3d.AddVolume(volume)

        self.setup_camera(self.renderer_3d)

        self.render_window_3d.Render()

        self.render_window_interactor_axial.RemoveObservers("LeftButtonPressEvent")
        self.render_window_interactor_coronal.RemoveObservers("LeftButtonPressEvent")
        self.render_window_interactor_sagittal.RemoveObservers("LeftButtonPressEvent")

        self.render_window_interactor_axial.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.render_window_interactor_axial.AddObserver("LeftButtonPressEvent", self.on_left_button_press, 114514.0)
        self.render_window_interactor_axial.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)

        self.render_window_interactor_coronal.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.render_window_interactor_coronal.AddObserver("LeftButtonPressEvent", self.on_left_button_press, 114514.0)
        self.render_window_interactor_coronal.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)

        self.render_window_interactor_sagittal.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.render_window_interactor_sagittal.AddObserver("LeftButtonPressEvent", self.on_left_button_press, 114514.0)
        self.render_window_interactor_sagittal.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release)

        # 测量两点距离的监听器
        self.render_window_interactor_axial.AddObserver("LeftButtonPressEvent", self.capture_point)
        self.render_window_interactor_coronal.AddObserver("LeftButtonPressEvent", self.capture_point)
        self.render_window_interactor_sagittal.AddObserver("LeftButtonPressEvent", self.capture_point)

        # 测量三点角度的监听器
        self.render_window_interactor_axial.AddObserver("LeftButtonPressEvent", self.capture_angle)
        self.render_window_interactor_coronal.AddObserver("LeftButtonPressEvent", self.capture_angle)
        self.render_window_interactor_sagittal.AddObserver("LeftButtonPressEvent", self.capture_angle)

        # 测量水平线角度的监听器
        self.render_window_interactor_axial.AddObserver("LeftButtonPressEvent", self.capture_angle_horizontal)
        self.render_window_interactor_coronal.AddObserver("LeftButtonPressEvent", self.capture_angle_horizontal)
        self.render_window_interactor_sagittal.AddObserver("LeftButtonPressEvent", self.capture_angle_horizontal)

        # 擦除的监听器
        self.render_window_interactor_axial.AddObserver("LeftButtonPressEvent", self.erase_marker)
        self.render_window_interactor_coronal.AddObserver("LeftButtonPressEvent", self.erase_marker)
        self.render_window_interactor_sagittal.AddObserver("LeftButtonPressEvent", self.erase_marker)

        self.add_right_click_zoom_handler(self.render_window_interactor_axial, self.axial_viewer)

        self.add_right_click_zoom_handler(self.render_window_interactor_coronal, self.coronal_viewer)

        self.add_right_click_zoom_handler(self.render_window_interactor_sagittal, self.sagittal_viewer)

        self.render_window_interactor_axial.AddObserver("RightButtonPressEvent", self.show_minimenu)
        self.render_window_interactor_coronal.AddObserver("RightButtonPressEvent", self.show_minimenu)
        self.render_window_interactor_sagittal.AddObserver("RightButtonPressEvent", self.show_minimenu)

        self.render_window_interactor_3d.AddObserver("LeftButtonPressEvent", self.projection_back)

        self.render_window_interactor_axial.Initialize()
        self.render_window_interactor_coronal.Initialize()
        self.render_window_interactor_sagittal.Initialize()
        self.render_window_interactor_3d.Initialize()

        # 添加红点和十字线
        self.add_marker_with_lines(middle_sagittal, middle_coronal, middle_axial)

        self.xy_plane_3d_actor = self.add_plane([0, 0, 1], self.center)
        self.xz_plane_3d_actor = self.add_plane([0, 1, 0], self.center)
        self.yz_plane_3d_actor = self.add_plane([1, 0, 0], self.center)

        self.renderer_3d.RemoveActor(self.xy_plane_3d_actor)
        self.renderer_3d.RemoveActor(self.yz_plane_3d_actor)
        self.renderer_3d.RemoveActor(self.xz_plane_3d_actor)

        self.render_window_3d.Render()
        # 显示3D映射
        if self.projection_3d:
            self.show_slice_position_in_3d()

        self.first_open = False


    def update_views(self):
        # 清除旧的标记和线条
        self.clear_marker_and_line()
        # 添加新的标记和线条
        x = self.x_input.value()
        y = self.y_input.value()
        z = self.z_input.value()

        self.add_marker_with_lines(x, y, z)

        self.axial_viewer.SetSlice(z)
        self.coronal_viewer.SetSlice(y)
        self.sagittal_viewer.SetSlice(x)

        self.axial_viewer.Render()
        self.coronal_viewer.Render()
        self.sagittal_viewer.Render()
        self.update_physical_position_label_map(x, y, z)
        if self.projection_3d:
            self.show_slice_position_in_3d()

    def add_marker_with_lines(self, x, y, z):

        for viewer, orientation in zip(
                [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer],
                ['axial', 'coronal', 'sagittal']):
            renderer = viewer.GetRenderer()

            # 创建红点
            point_source = vtk.vtkPointSource()
            point_source.SetCenter(x, y, z)
            point_source.SetNumberOfPoints(1)

            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetRadius(3.0)

            glyph = vtk.vtkGlyph3D()
            glyph.SetSourceConnection(sphere_source.GetOutputPort())
            glyph.SetInputConnection(point_source.GetOutputPort())

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 1, 0)  # 黄色

            renderer.AddActor(actor)
            self.markers.append(actor)

            # 创建X轴平行线
            line_source_horizontal = vtk.vtkLineSource()
            if orientation == 'axial':
                line_source_horizontal.SetPoint1(0, y, z)
                line_source_horizontal.SetPoint2(self.width - 1, y, z)
            elif orientation == 'coronal':
                line_source_horizontal.SetPoint1(0, y, z)
                line_source_horizontal.SetPoint2(self.width - 1, y, z)
            elif orientation == 'sagittal':
                line_source_horizontal.SetPoint1(x, 0, z)
                line_source_horizontal.SetPoint2(x, self.height - 1, z)

            line_mapper_horizontal = vtk.vtkPolyDataMapper()
            line_mapper_horizontal.SetInputConnection(line_source_horizontal.GetOutputPort())

            line_actor_horizontal = vtk.vtkActor()
            line_actor_horizontal.SetMapper(line_mapper_horizontal)
            if self.picking:
                line_actor_horizontal.GetProperty().SetColor(0, 0, 1)  # 蓝色
            else:
                line_actor_horizontal.GetProperty().SetColor(0, 1, 0)  # 绿色

            renderer.AddActor(line_actor_horizontal)
            self.lines.append(line_actor_horizontal)

            # 创建Y轴平行线
            line_source_vertical = vtk.vtkLineSource()
            if orientation == 'axial':
                line_source_vertical.SetPoint1(x, 0, z)
                line_source_vertical.SetPoint2(x, self.height - 1, z)
            elif orientation == 'coronal':
                line_source_vertical.SetPoint1(x, y, 0)
                line_source_vertical.SetPoint2(x, y, self.depth - 1)
            elif orientation == 'sagittal':
                line_source_vertical.SetPoint1(x, y, 0)
                line_source_vertical.SetPoint2(x, y, self.depth - 1)

            line_mapper_vertical = vtk.vtkPolyDataMapper()
            line_mapper_vertical.SetInputConnection(line_source_vertical.GetOutputPort())

            line_actor_vertical = vtk.vtkActor()
            line_actor_vertical.SetMapper(line_mapper_vertical)
            if self.picking:
                line_actor_vertical.GetProperty().SetColor(0, 0, 1)  # 蓝色
            else:
                line_actor_vertical.GetProperty().SetColor(0, 1, 0)  # 绿色

            renderer.AddActor(line_actor_vertical)
            self.lines.append(line_actor_vertical)

            viewer.Render()

    # 检测鼠标是否接近辅助线,当鼠标按下并拖动时，更新对应的坐标值并刷新视图
    def on_mouse_move(self, obj, event):
        if not (self.marking) and not (self.erasing) and not (self.measuring) and not (self.measuring_angle) and not (
        self.picking):
            interactor = obj
            mouse_pos = interactor.GetEventPosition()
            picker = vtk.vtkWorldPointPicker()
            picker.Pick(mouse_pos[0], mouse_pos[1], 0, interactor.GetRenderWindow().GetRenderers().GetFirstRenderer())
            mouse_world_pos = picker.GetPickPosition()
            x, y, z = mouse_world_pos

            # 假设axial面上的横轴和纵轴的位置分别是self.x_line_pos和self.y_line_pos
            x_line_pos = self.x_input.value()
            y_line_pos = self.y_input.value()
            z_line_pos = self.z_input.value()

            # 定义一个阈值范围
            threshold = 10

            if obj == self.render_window_interactor_axial:
                # 检测鼠标位置是否接近横轴或纵轴
                if abs(y - y_line_pos) < threshold:
                    QApplication.setOverrideCursor(Qt.SizeVerCursor)  # 竖直方向的手形光标
                    self.hovered_line = 'y'
                elif abs(x - x_line_pos) < threshold:
                    QApplication.setOverrideCursor(Qt.SizeHorCursor)  # 水平方向的手形光标
                    self.hovered_line = 'x'
                else:
                    QApplication.restoreOverrideCursor()
                    self.hovered_line = None
            elif obj == self.render_window_interactor_coronal:
                # 检测鼠标位置是否接近横轴或纵轴
                if abs(z - z_line_pos) < threshold:
                    QApplication.setOverrideCursor(Qt.SizeVerCursor)  # 竖直方向的手形光标
                    self.hovered_line = 'z'
                elif abs(x - x_line_pos) < threshold:
                    QApplication.setOverrideCursor(Qt.SizeHorCursor)  # 水平方向的手形光标
                    self.hovered_line = 'x'
                else:
                    QApplication.restoreOverrideCursor()
                    self.hovered_line = None
            elif obj == self.render_window_interactor_sagittal:
                # 检测鼠标位置是否接近横轴或纵轴
                if abs(z - z_line_pos) < threshold:
                    QApplication.setOverrideCursor(Qt.SizeVerCursor)  # 竖直方向的手形光标
                    self.hovered_line = 'z'
                elif abs(y - y_line_pos) < threshold:
                    QApplication.setOverrideCursor(Qt.SizeHorCursor)  # 水平方向的手形光标
                    self.hovered_line = 'y'
                else:
                    QApplication.restoreOverrideCursor()
                    self.hovered_line = None

            if self.mouse_pressed and self.dragging_line:
                if self.dragging_line == 'y':
                    self.y_input.setValue(y)
                    self.update_views()
                elif self.dragging_line == 'x':
                    self.x_input.setValue(x)
                    self.update_views()
                elif self.dragging_line == 'z':
                    self.z_input.setValue(z)
                    self.update_views()

    def on_left_button_press(self, obj, event):
        if self.hovered_line:
            self.mouse_pressed = True
            self.dragging_line = self.hovered_line

    def on_left_button_release(self, obj, event):
        self.mouse_pressed = False
        self.dragging_line = None

    def clear_markers(self):
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            renderer = viewer.GetRenderer()
            for marker in self.markers:
                renderer.RemoveActor(marker)
            viewer.Render()
        self.markers = []

    def clear_lines(self):
        for viewer in [self.axial_viewer, self.coronal_viewer, self.sagittal_viewer]:
            renderer = viewer.GetRenderer()
            for marker in self.lines:
                renderer.RemoveActor(marker)
            viewer.Render()
        self.lines = []

    def clear_marker_and_line(self):
        self.clear_markers()
        self.clear_lines()

    def update_inputs_from_viewer(self, obj, event):
        self.z_input.setValue(self.axial_viewer.GetSlice())
        self.y_input.setValue(self.coronal_viewer.GetSlice())
        self.x_input.setValue(self.sagittal_viewer.GetSlice())
        self.update_physical_position_label_map(self.x_input.value(), self.y_input.value(), self.z_input.value())

    def update_inputs_from_image(self):
        self.z_input.setValue(self.axial_viewer.GetSlice())
        self.y_input.setValue(self.coronal_viewer.GetSlice())
        self.x_input.setValue(self.sagittal_viewer.GetSlice())

    def picking_switch(self):
        # 设置交互器捕获鼠标事件
        if self.picking:
            # 恢复正常的鼠标样式并停止捕获事件
            QApplication.restoreOverrideCursor()
            self.picking = False
            self.pick_label.setText("Pick point is OFF")
            self.pick_label.setStyleSheet("color: black;")
            self.update_views()
        else:
            self.render_window_interactor_axial.AddObserver("LeftButtonPressEvent", self.pick_point)
            self.render_window_interactor_coronal.AddObserver("LeftButtonPressEvent", self.pick_point)
            self.render_window_interactor_sagittal.AddObserver("LeftButtonPressEvent", self.pick_point)
            self.picking = True
            self.update_views()
            QApplication.setOverrideCursor(Qt.CrossCursor)
            self.pick_label.setText("Pick point is ON")
            self.pick_label.setStyleSheet("color: green;")

    def pick_point(self, obj, event):
        if self.picking:
            click_pos = obj.GetEventPosition()
            picker = vtk.vtkCellPicker()

            # 确保渲染器存在
            renderer = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
            if renderer is not None:
                self.save_state_snapshot()
                picker.Pick(click_pos[0], click_pos[1], 0, renderer)
                world_pos = picker.GetPickPosition()
                self.coord_label.setText(f"Coordinates: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})")
                if obj == self.render_window_interactor_axial:
                    self.x_input.setValue(round(world_pos[0]))
                    self.y_input.setValue(round(world_pos[1]))
                elif obj == self.render_window_interactor_coronal:
                    self.x_input.setValue(round(world_pos[0]))
                    self.z_input.setValue(round(world_pos[2]))
                elif obj == self.render_window_interactor_sagittal:
                    self.y_input.setValue(round(world_pos[1]))
                    self.z_input.setValue(round(world_pos[2]))
                self.update_physical_position_label_map(world_pos[0], world_pos[1], world_pos[2])

    def rotate_x(self, value):
        self.save_state_snapshot()
        self.transform_x.Identity()
        self.transform_x.Translate(self.center)
        self.transform_x.RotateX(value)
        self.transform_x.Translate(-self.center[0], -self.center[1], -self.center[2])
        self.rotate_x_input.setValue(value)
        self.update_reslice()

    def rotate_y(self, value):
        self.save_state_snapshot()
        self.transform_y.Identity()
        self.transform_y.Translate(self.center)
        self.transform_y.RotateY(value)
        self.transform_y.Translate(-self.center[0], -self.center[1], -self.center[2])
        self.rotate_y_input.setValue(value)
        self.update_reslice()

    def rotate_z(self, value):
        self.save_state_snapshot()
        self.transform_z.Identity()
        self.transform_z.Translate(self.center)
        self.transform_z.RotateZ(value)
        self.transform_z.Translate(-self.center[0], -self.center[1], -self.center[2])
        self.rotate_z_input.setValue(value)
        self.update_reslice()

    def update_rotate_x(self):
        value = self.rotate_x_input.value()
        if value == 360:
            value = 0
        elif value == -1:
            value = 359
        self.rotate_x(value)
        # self.rotate_x_dial.setValue(value)
        self.rotate_x_input.setValue(value)

    def update_rotate_y(self):
        value = self.rotate_y_input.value()
        if value == 360:
            value = 0
        elif value == -1:
            value = 359
        self.rotate_y(value)
        # self.rotate_y_dial.setValue(value)
        self.rotate_y_input.setValue(value)

    def update_rotate_z(self):
        value = self.rotate_z_input.value()
        if value == 360:
            value = 0
        elif value == -1:
            value = 359
        self.rotate_z(value)
        # self.rotate_z_dial.setValue(value)
        self.rotate_z_input.setValue(value)

    def update_reslice(self):
        combined_transform = vtk.vtkTransform()
        combined_transform.Concatenate(self.transform_z)
        combined_transform.Concatenate(self.transform_y)
        combined_transform.Concatenate(self.transform_x)
        self.reslice.SetResliceAxes(combined_transform.GetMatrix())
        self.axial_viewer.Render()
        self.coronal_viewer.Render()
        self.sagittal_viewer.Render()

    # 这个方法接受的是转前的世界坐标，返回的是转后的世界坐标,这里是以center为中心，欧拉角
    def calculate_position_in_key_coordinates(self, x, y, z, angle_x, angle_y, angle_z):
        # 输入x,y,z,角度x, 角度y, 角度z,xyz为点的坐标，角度为现在视图所处角度;
        # 计算某点在关键点坐标系方向下的坐标，注意这是切片的序列号，还不是最终的物理位置
        point = self.rotate_coordinate_plus(x, y, z, -angle_x, -angle_y, -angle_z, True)
        # 某点处在已经旋转过的画面下。现在先将其根据角度逆转回到原始坐标系，再将其转到关键点坐标系
        # 与三维映射毫不干涉
        pos = self.rotate_coordinate_plus(point[0], point[1], point[2],
                                          self.euler_angles[0],
                                          self.euler_angles[1],
                                          self.euler_angles[2])
        return pos

    # 这个方法接受的是转前的世界坐标，返回的是转后的世界坐标,这里是以center为中心，图片欧拉角
    def calculate_position_in_key_coordinates_map(self, x, y, z, angle_x, angle_y, angle_z):
        # 输入x,y,z,角度x, 角度y, 角度z,xyz为点的坐标，角度为现在视图所处角度;
        # 计算某点在关键点坐标系方向下的坐标，注意这是切片的序列号，还不是最终的物理位置
        point = self.rotate_coordinate_plus(x, y, z, -angle_x, -angle_y, -angle_z, True)
        # 某点处在已经旋转过的画面下。现在先将其根据角度逆转回到原始坐标系，再将其转到关键点坐标系
        # 与三维映射毫不干涉
        pos = self.rotate_coordinate_plus(point[0], point[1], point[2],
                                          self.euler_angles_map[0],
                                          self.euler_angles_map[1],
                                          self.euler_angles_map[2])
        pos = [round(p) for p in pos]
        return pos

    # 这个方法接受的是转前世界坐标，返回的是物理坐标
    def update_physical_position_label_plus(self, x, y, z, angle_x=0, angle_y=0, angle_z=0, display=True):
        point = self.rotate_coordinate(x, y, z, -angle_x, -angle_y, -angle_z)
        pos = self.rotate_coordinate_plus(point[0], point[1], point[2],
                                          self.euler_angles[0],
                                          self.euler_angles[1],
                                          self.euler_angles[2])
        slice_pos = np.array([pos[0], pos[1], pos[2]]) - self.origin_physical
        position = [x * self.slice_thickness for x in slice_pos]

        position[2] = -position[2]
        position[0] = -position[0]

        pos_plus = (position[0], position[1], position[2])
        return pos_plus

    # 这个方法接受的是转后图像上一点世界坐标，返回的是物理坐标
    def update_physical_position_label_map(self, x, y, z, display=True):
        slice_pos = np.array([x, y, z]) - self.origin_physical_map
        position = [x * self.slice_thickness for x in slice_pos]
        if display:
            self.physical_position_label.setText(
                f"Physical Position: ({position[1]:.2f}, {position[0]:.2f}, {position[2]:.2f})")

        pos = (position[0], position[1], position[2])
        return pos

    # 这个方法接受的是转后图像上一点世界坐标，返回的是原世界坐标
    def update_world_position_label_map(self, x, y, z):
        pos2 = self.rotate_coordinate_plus(x, y, z,
                                           -self.euler_angles_map[0],
                                           -self.euler_angles_map[1],
                                           -self.euler_angles_map[2])
        return pos2

    def closeEvent(self, event):
        # Finalize all render windows
        self.render_window_axial.Finalize()
        self.render_window_coronal.Finalize()
        self.render_window_sagittal.Finalize()
        self.render_window_3d.Finalize()

        # Terminate all interactors
        self.render_window_interactor_axial.TerminateApp()
        self.render_window_interactor_coronal.TerminateApp()
        self.render_window_interactor_sagittal.TerminateApp()
        self.render_window_interactor_3d.TerminateApp()

        # Super closeEvent to ensure the window is closed
        super(MainWindow, self).closeEvent(event)

    def measure_two_points(self):
        self.measuring = True
        self.point1 = None
        self.point2 = None
        self.click = 0
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def capture_point(self, obj, event):
        if self.measuring:
            click_pos = obj.GetEventPosition()
            picker = vtk.vtkCellPicker()
            renderer = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
            if obj == self.render_window_interactor_axial:
                self.current_viewer = self.axial_viewer
            elif obj == self.render_window_interactor_coronal:
                self.current_viewer = self.coronal_viewer
            elif obj == self.render_window_interactor_sagittal:
                self.current_viewer = self.sagittal_viewer
            if renderer is not None:
                picker.Pick(click_pos[0], click_pos[1], 0, renderer)
                world_pos = picker.GetPickPosition()

                if self.click == 0:
                    self.point1 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.click += 1
                else:
                    self.point2 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.draw_line_and_measure(self.point1, self.point2, True)
                    self.measuring = False
                    self.click = 0
                    self.point1 = None
                    self.point2 = None
                    self.current_viewer = None
                    QApplication.restoreOverrideCursor()

    def measure_one_angle(self):
        self.measuring_angle = True
        self.measuring = False
        self.picking = False
        self.marking = False
        self.point1 = None
        self.point2 = None
        self.point3 = None
        self.click = 0
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def measure_angle_horizontal(self):
        self.measuring_horizontal_angle = True
        self.measuring = False
        self.picking = False
        self.marking = False
        self.point1 = None
        self.point2 = None
        self.point3 = None
        self.click = 0
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def capture_angle(self, obj, event):
        if self.measuring_angle:
            click_pos = obj.GetEventPosition()
            picker = vtk.vtkCellPicker()
            renderer = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
            if obj == self.render_window_interactor_axial:
                self.current_viewer = self.axial_viewer
            elif obj == self.render_window_interactor_coronal:
                self.current_viewer = self.coronal_viewer
            elif obj == self.render_window_interactor_sagittal:
                self.current_viewer = self.sagittal_viewer
            if renderer is not None:
                picker.Pick(click_pos[0], click_pos[1], 0, renderer)
                world_pos = picker.GetPickPosition()
                if self.click == 0:
                    self.point1 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.click += 1
                elif self.click == 1:
                    self.point2 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.draw_line_and_measure(self.point1, self.point2, True)
                    self.click += 1
                else:
                    self.point3 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.draw_line_and_measure(self.point2, self.point3, True)
                    self.label_angle(self.point1, self.point2, self.point3)
                    self.measuring_angle = False
                    self.click = 0
                    self.point1 = None
                    self.point2 = None
                    self.current_viewer = None
                    QApplication.restoreOverrideCursor()

    def capture_angle_horizontal(self, obj, event):
        if self.measuring_horizontal_angle:
            click_pos = obj.GetEventPosition()
            picker = vtk.vtkCellPicker()
            renderer = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
            if obj == self.render_window_interactor_axial:
                self.current_viewer = self.axial_viewer
            elif obj == self.render_window_interactor_coronal:
                self.current_viewer = self.coronal_viewer
            elif obj == self.render_window_interactor_sagittal:
                self.current_viewer = self.sagittal_viewer
            if renderer is not None:
                picker.Pick(click_pos[0], click_pos[1], 0, renderer)
                world_pos = picker.GetPickPosition()
                if self.click == 0:
                    self.point1 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.click += 1
                else:
                    self.point2 = world_pos
                    self.add_marker(self.current_viewer, world_pos)
                    self.draw_line_and_measure(self.point1, self.point2, True)

                    self.point3 = world_pos

                    if obj == self.render_window_interactor_axial:
                        self.point3 = self.rotate_coordinate(self.width, world_pos[1], world_pos[2], 0, 0,
                                                             self.rotate_z_input.value())
                    elif obj == self.render_window_interactor_coronal:
                        self.point3 = self.rotate_coordinate(self.width, world_pos[1], world_pos[2], 0, 0,
                                                             self.rotate_z_input.value())
                    elif obj == self.render_window_interactor_sagittal:
                        self.point3 = self.rotate_coordinate(world_pos[0], self.height, world_pos[2], 0, 0,
                                                             self.rotate_z_input.value())

                    self.add_marker(self.current_viewer, world_pos)
                    self.draw_line_and_measure(self.point2, self.point3, False)
                    self.label_angle(self.point1, self.point2, self.point3)
                    self.measuring_horizontal_angle = False
                    self.click = 0
                    self.point1 = None
                    self.point2 = None
                    self.point3 = None
                    self.current_viewer = None
                    QApplication.restoreOverrideCursor()

    def label_angle(self, point1, point2, point3):
        # 计算角度
        angle = self.calculate_angle(point1, point2, point3)

        text_source = vtk.vtkTextSource()

        text_source.SetText(f"{angle:.2f}°")
        text_source.SetBackgroundColor(1.0, 1.0, 1.0)
        text_source.SetForegroundColor(1.0, 0.0, 0.0)
        text_source.Update()

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        text_actor = vtk.vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetScale(0.5, 0.5, 0.5)
        mid_point = (np.array(point2) + np.array(point2)) / 2
        if self.current_viewer == self.axial_viewer:
            text_actor.SetPosition(mid_point[0] + 1, mid_point[1] + 1, self.z_input.value() + 1)
        elif self.current_viewer == self.coronal_viewer:
            text_actor.SetPosition(mid_point[0] + 1, self.y_input.value() - 1, mid_point[2] + 1)
        else:
            text_actor.SetPosition(self.x_input.value() + 1, mid_point[1] + 1, mid_point[2] + 1)
        text_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        text_actor.SetCamera(self.current_viewer.GetRenderer().GetActiveCamera())

        self.current_viewer.GetRenderer().AddActor(text_actor)
        self.current_viewer.Render()

        angle_name = f"Angle {len(self.marked_points) + 1}"

        self.angles.append((angle_name, angle))

        if self.measuring_angle:
            self.last_angle_label.setText(f"Last Angle: {angle:.2f} °")
        elif self.measuring_horizontal_angle:
            self.last_h_angle_label.setText(f"Last Horizontal Angle: {angle:.2f} °")

    def calculate_angle(self, point1, point2, point3):
        # 将点转换为 NumPy 数组
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # 计算向量
        v1 = p1 - p2
        v2 = p3 - p2

        # 计算向量的单位向量
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)

        # 计算点积和角度
        dot_product = np.dot(v1_u, v2_u)
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def add_marker(self, viewer, world_pos):
        point_source = vtk.vtkPointSource()
        point_source.SetCenter(world_pos)
        point_source.SetNumberOfPoints(1)

        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(self.marker_radius)

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere_source.GetOutputPort())
        glyph.SetInputConnection(point_source.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # 红色

        # self.mark_point_markers.append(actor)

        viewer.GetRenderer().AddActor(actor)
        viewer.Render()

        actor_center = actor.GetCenter()

        return actor_center
        # 红点的实际中心与点击位置有非常细微的差距，推测是由于浮点计算的精度问题导致，目前未找到解决办法
        # 所以最终选择将红点实际中心返还并加入表格中，以确保图像中红点中心与表格对齐，特此注明

    def add_plane(self, normal, point):
        # 创建平面源
        point = np.array((- point[0], point[1], point[2]))
        normal = np.array(normal)
        plane_source = vtk.vtkPlaneSource()

        # 设置平面的中心和法向量
        plane_source.SetOrigin(0, 0, 0)
        plane_source.SetPoint1(768, 0, 0)
        plane_source.SetPoint2(0, 768, 0)

        plane_source.SetCenter(point)
        plane_source.SetNormal(normal)

        # 设置平面的尺寸
        plane_source.SetXResolution(50)
        plane_source.SetYResolution(50)

        # 更新平面源
        plane_source.Update()

        # 映射平面数据
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane_source.GetOutputPort())

        # 创建平面演员
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)  # 设置平面颜色，绿色

        # 添加平面到渲染器
        self.renderer_3d.AddActor(actor)

        self.render_window_3d.Render()
        return actor

    def draw_line_and_measure(self, point1, point2, drawing):
        print(f'调用')
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(point1)
        line_source.SetPoint2(point2)

        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(line_source.GetOutputPort())

        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0, 1, 0)  # 绿色

        if drawing:
            self.current_viewer.GetRenderer().AddActor(line_actor)

        # 计算距离
        if self.measuring and not self.measuring_angle and not self.measuring_horizontal_angle:
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            distance *= self.slice_thickness
            self.display_distance(self.current_viewer, point1, point2, distance)
            distance_name = f"Distance {len(self.distances) + 1}"
            self.distances.append((distance_name, distance))
            self.last_distance_label.setText(f"Last Distance: {distance:.2f} mm")


        self.current_viewer.Render()

    def display_distance(self, viewer, point1, point2, distance):
        text_source = vtk.vtkTextSource()
        text_source.SetText(f"{distance:.2f} mm")
        text_source.SetBackgroundColor(1.0, 1.0, 1.0)
        text_source.SetForegroundColor(1.0, 0.0, 0.0)
        text_source.Update()

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        text_actor = vtk.vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetScale(0.5, 0.5, 0.5)
        mid_point = (np.array(point1) + np.array(point2)) / 2
        if viewer == self.axial_viewer:
            text_actor.SetPosition(mid_point[0] + 1, mid_point[1] + 1, self.z_input.value() + 1)
        elif viewer == self.coronal_viewer:
            text_actor.SetPosition(mid_point[0] + 1, self.y_input.value() - 1, mid_point[2] + 1)
        else:
            text_actor.SetPosition(self.x_input.value() + 1, mid_point[1] + 1, mid_point[2] + 1)
        text_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        text_actor.SetCamera(viewer.GetRenderer().GetActiveCamera())

        viewer.GetRenderer().AddActor(text_actor)
        viewer.Render()

    # 求SR以center为中心转后的世界坐标，欧拉角
    def set_physical_origin(self):
        self.origin_physical = self.calculate_position_in_key_coordinates(self.SR[1][0],
                                                                          self.SR[1][1],
                                                                          self.SR[1][2],
                                                                          self.SR[2][0],
                                                                          self.SR[2][1],
                                                                          self.SR[2][2])

    # 求SR以center为中心转后的世界坐标，图片欧拉角
    def set_physical_origin_map(self):
        self.origin_physical_map = self.calculate_position_in_key_coordinates_map(self.SR[1][0],
                                                                                  self.SR[1][1],
                                                                                  self.SR[1][2],
                                                                                  self.SR[2][0],
                                                                                  self.SR[2][1],
                                                                                  self.SR[2][2])

    # 以SR为中心进行旋转
    def rotate_coordinate(self, x, y, z, angle_x, angle_y, angle_z, reverse_turn=False):
        """旋转坐标点"""

        # 将角度从度转换为弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)

        # 绕X、Y、Z轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z
        R = R.transpose()

        # 原始点向量
        original_vector = np.array([x, y, z])

        # 将点平移到原点
        translated_point = original_vector - self.origin_world

        # 旋转点
        rotated_point = R @ translated_point

        # 将点平移回中心
        new_point = rotated_point + self.origin_world

        return new_point

    # 以图像中心点为中心进行旋转
    def rotate_coordinate_plus(self, x, y, z, angle_x, angle_y, angle_z, reverse_turn=False):
        """旋转坐标点"""

        # 将角度从度转换为弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)

        # 绕X、Y、Z轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z
        R = R.transpose()

        # 原始点向量
        original_vector = np.array([x, y, z])

        # 将点平移到原点
        translated_point = original_vector - self.center

        # 旋转点
        rotated_point = R @ translated_point

        # 将点平移回中心
        new_point = rotated_point + self.center

        return new_point

    def switch_projection_back(self):
        if self.projection_3d_2d:
            self.projection_3d_2d = False
            self.projection_3d_2d_label.setText("3映2：关闭")
            self.projection_3d_2d_label.setStyleSheet("color: black;")
        else:
            self.projection_3d_2d = True
            self.projection_3d_2d_label.setText("3映2：开启")
            self.projection_3d_2d_label.setStyleSheet("color: green;")

    def projection_back(self, obj, event):
        if self.projection_3d_2d:
            click_pos = obj.GetEventPosition()
            picker = vtk.vtkCellPicker()
            renderer = obj.GetRenderWindow().GetRenderers().GetFirstRenderer()
            if renderer is not None:
                picker.Pick(click_pos[0], click_pos[1], 0, renderer)
                world_pos = picker.GetPickPosition()
                pos = (-round(world_pos[0]), round(world_pos[1]), round(world_pos[2]))
                pos = self.rotate_coordinate(pos[0], pos[1], pos[2],
                                             -self.rotate_x_input.value(),
                                             -self.rotate_y_input.value(),
                                             -self.rotate_z_input.value())
                self.x_input.setValue(round(pos[0]))
                self.y_input.setValue(round(pos[1]))
                self.z_input.setValue(round(pos[2]))

    def coordinate_planes_switch(self):
        # 切换到矢状面视图
        self.current_viewer = self.sagittal_viewer
        self.draw_line_and_measure(self.AODA[1], self.ANS[1], True)

        if self.coords_plane_display:
            # 移除固定的 Actor 实例（如果存在）
            self.safe_remove_actor(self.xy_plane_3d_actor)
            self.safe_remove_actor(self.yz_plane_3d_actor)
            self.safe_remove_actor(self.xz_plane_3d_actor)
            self.coords_plane_display = False
        else:
            # 添加固定的 Actor 实例（如果未添加）
            self.safe_add_actor(self.xy_plane_3d_actor)
            self.safe_add_actor(self.yz_plane_3d_actor)
            self.safe_add_actor(self.xz_plane_3d_actor)
            self.coords_plane_display = True

        self.render_window_3d.Render()

    # 辅助方法：安全添加/移除 Actor
    def safe_remove_actor(self, actor):
        if actor is None:
            return
        # 获取渲染器中所有 Actor 的集合
        actors = self.renderer_3d.GetActors()
        actors.InitTraversal()
        # 遍历检查是否存在目标 Actor
        for _ in range(actors.GetNumberOfItems()):
            current_actor = actors.GetNextActor()
            if current_actor == actor:
                self.renderer_3d.RemoveActor(actor)
                break  # 找到后立即退出循环

    def safe_add_actor(self, actor):
        if actor is None:
            return
        # 同样遍历检查是否已存在
        actors = self.renderer_3d.GetActors()
        actors.InitTraversal()
        exists = False
        for _ in range(actors.GetNumberOfItems()):
            current_actor = actors.GetNextActor()
            if current_actor == actor:
                exists = True
                break
        if not exists:
            self.renderer_3d.AddActor(actor)

    def mirror_vector_yz(self, vector):
        """
        Mirror a vector across the YZ plane.

        Parameters:
        vector (np.array or list): The input vector [x, y, z].

        Returns:
        np.array: The mirrored vector [-x, y, z].
        """
        vector = np.array(vector)
        mirrored_vector = np.array([-vector[0], vector[1], vector[2]])
        return mirrored_vector

    def set_coordinate_system(self):
        if self.AODA is not None and self.ANS is not None and self.HtR is not None and self.HtL is not None and self.SR is not None:
            # 首先将所有点换算到统一坐标系
            points = []  # (angle_x, angle_y, angle_z)

            key_points = [self.AODA, self.ANS, self.HtR, self.HtL, self.SR]
            print(f'开始建立坐标系，本次关键点坐标为：{key_points}')
            print(f'center:{self.center}')

            self.origin_world = [self.SR[1][0], self.SR[1][1], self.SR[1][2]]

            for (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in key_points:
                pos = self.rotate_coordinate(x, y, z,
                                             angle_x,
                                             angle_y,
                                             angle_z)
                points.append((pos[0], pos[1], pos[2]))

            vector_AB = np.array(
                [points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2]])
            vector_CD = np.array(
                [points[3][0] - points[2][0], points[3][1] - points[2][1], points[3][2] - points[2][2]])

            # print(vector_AB, vector_CD)

            vector_axial = self.normalize_vector(np.cross(vector_AB, vector_CD))  # 水平面的法向量,新z轴
            vector_coronal = self.normalize_vector(np.cross(vector_CD, vector_axial))  # 冠状面的法向量，新y轴
            vector_sagittal = self.normalize_vector(np.cross(vector_coronal, vector_axial))  # 矢状面的法向量，新x轴

            # print(vector_sagittal)
            # print(vector_coronal)
            # print(vector_axial)

            # 计算旋转矩阵
            rotation_matrix = self.rotation_matrix_from_vectors(vector_sagittal, vector_coronal, vector_axial)
            # print(f"rotation_matrix\n{rotation_matrix}")

            # 从旋转矩阵计算欧拉角
            euler_angles = self.euler_angles_from_rotation_matrix(rotation_matrix)
            self.euler_angles = euler_angles
            # print(f"euler_angles\n{self.euler_angles}")

            self.euler_angles_map[0] = (180 + self.euler_angles[0]) % 360
            self.euler_angles_map[1] = -(self.euler_angles[1]) % 360
            self.euler_angles_map[2] = -(self.euler_angles[2]) % 360
            self.euler_angles_map[2] = 180 - self.euler_angles[2]
            # print(f'euler_angles_map\n{self.euler_angles_map}')

            self.set_physical_origin()
            # print(f'physical_origin\n{self.origin_physical}')
            self.set_physical_origin_map()
            # print(f'self.origin_physical_map:\n{self.origin_physical_map}')

            # 设置坐标系后，将现在视图调至坐标原点，将现在视图角度校正为坐标系方向
            self.x_input.setValue(self.origin_physical_map[0])
            self.y_input.setValue(self.origin_physical_map[1])
            self.z_input.setValue(self.origin_physical_map[2])

            self.rotate_x_input.setValue(self.euler_angles_map[0])
            self.rotate_y_input.setValue(self.euler_angles_map[1])
            self.rotate_z_input.setValue(self.euler_angles_map[2])

            if self.coords_plane_display:
                self.coordinate_planes_switch()

            vector_axial = self.mirror_vector_yz(vector_axial)
            vector_coronal = self.mirror_vector_yz(vector_coronal)
            vector_sagittal = self.mirror_vector_yz(vector_sagittal)

            # 在创建新平面之前，清理旧平面
            self.safe_remove_actor(self.xy_plane_3d_actor)
            self.safe_remove_actor(self.yz_plane_3d_actor)
            self.safe_remove_actor(self.xz_plane_3d_actor)

            # 创建新平面
            self.xy_plane_3d_actor = self.add_plane(vector_axial, points[4])
            self.yz_plane_3d_actor = self.add_plane(vector_coronal, points[4])
            self.xz_plane_3d_actor = self.add_plane(vector_sagittal, points[4])

            self.key_points.clear()

            for (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in key_points:
                slice_pos = (x, y, z)
                angles = (angle_x, angle_y, angle_z)
                physicals = self.update_physical_position_label_plus(x, y, z, angle_x, angle_y, angle_z)
                PT = (name, slice_pos, angles, physicals)
                if name == "AODA":
                    self.AODA = PT
                elif name == "ANS":
                    self.ANS = PT
                elif name == "HtR":
                    self.HtR = PT
                elif name == "HtL":
                    self.HtL = PT
                elif name == "SR":
                    self.SR = PT
                self.key_points.append(PT)

            self.update_physical_position_label_map(self.x_input.value(),self.y_input.value(),self.z_input.value())

            print("key_points\n", self.key_points)

            self.dicom_viewers[self.current_viewer_index].system = 1  # 建立坐标系的标志位



        else:
            QMessageBox.information(self, "ERROR", "请确保所有关键点都已经定义（在View中可以查看）")

    def normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def rotation_matrix_from_vectors(self, v1, v2, v3):
        # 将归一化向量作为旋转矩阵的列向量
        rotation_matrix = np.column_stack((v1, v2, v3))
        return rotation_matrix

    def euler_angles_from_rotation_matrix(self, matrix):
        """从旋转矩阵计算欧拉角"""
        # 计算y角
        y = np.arctan2(matrix[0, 2], np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))

        # 检查是否接近万向锁情况
        if np.abs(y - np.pi / 2) < 1e-6:
            # 万向锁情况，y = 90度
            print("警告：检测到万向锁情况（y = 90度）")
            z = 0
            x = np.arctan2(matrix[1, 0], matrix[1, 1])
        elif np.abs(y + np.pi / 2) < 1e-6:
            # 万向锁情况，y = -90度
            print("警告：检测到万向锁情况（y = -90度）")
            z = 0
            x = np.arctan2(-matrix[1, 0], -matrix[1, 1])
        else:
            # 一般情况
            z = np.arctan2(-matrix[0, 1], matrix[0, 0])
            x = np.arctan2(-matrix[1, 2], matrix[2, 2])

        # 将弧度转换为角度
        x = np.degrees(x) % 360
        y = np.degrees(y) % 360
        z = np.degrees(z) % 360

        return x, y, z


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    app.shutdown()
