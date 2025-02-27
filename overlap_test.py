import sys
import vtk
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QPushButton
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
import itk

class MouseInteractorStyle(vtk.vtkInteractorStyleImage):
    def __init__(self, renderer, image_handlers, current_image_index):
        super().__init__()
        self.renderer = renderer
        self.image_handlers = image_handlers
        self.current_image_index = current_image_index
        self.dragging = False
        self.last_mouse_position = None

    def get_current_image_actor(self):
        """
        获取当前处理的图片的 vtkImageActor
        """
        return self.image_handlers[self.current_image_index].get_image_actor()

    def left_button_press_event(self, obj, event):
        self.last_mouse_position = self.GetInteractor().GetEventPosition()
        self.dragging = True

    def left_button_release_event(self, obj, event):
        self.dragging = False

    def mouse_move_event(self, obj, event):
        if self.dragging:
            mouse_position = self.GetInteractor().GetEventPosition()
            dx = mouse_position[0] - self.last_mouse_position[0]
            dy = mouse_position[1] - self.last_mouse_position[1]
            self.last_mouse_position = mouse_position

            current_actor = self.get_current_image_actor()
            current_position = current_actor.GetPosition()
            current_actor.SetPosition(current_position[0] + dx, current_position[1] + dy, 0)
            self.GetInteractor().GetRenderWindow().Render()


class ImageHandler:
    def __init__(self, vtk_image, color):
        self.vtk_image = vtk_image
        self.color = color  # 颜色参数，例如 (1, 0, 0) 表示红色
        self.image_actor = None
        self.slice_orientation = 'coronal'  # 默认视图方向
        self.create_image_actor()

    def create_image_actor(self):
        """
        创建 vtkImageActor 并绑定到当前图片
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self.vtk_image)
        reslice.SetOutputDimensionality(2)

        if self.slice_orientation == 'coronal':
            reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, 1, 0)
            reslice.SetResliceAxesOrigin(0, self.vtk_image.GetDimensions()[1] // 2, 0)
        # 其他视图方向同理...

        reslice.SetInterpolationModeToLinear()

        # 创建颜色映射
        color_map = vtk.vtkImageMapToColors()
        color_map.SetInputConnection(reslice.GetOutputPort())
        color_map.SetLookupTable(self.create_color_lut(self.color))

        self.image_actor = vtk.vtkImageActor()
        self.image_actor.GetMapper().SetInputConnection(color_map.GetOutputPort())

    def create_color_lut(self, color):
        """
        创建颜色查找表 (Lookup Table)
        :param color: 目标颜色，例如 (1, 0, 0) 表示红色
        :return: vtkLookupTable
        """
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(0, 255)
        for i in range(256):
            lut.SetTableValue(i, color[0], color[1], color[2], i / 255.0)  # 设置透明度和颜色
        lut.Build()
        return lut

    def set_opacity(self, opacity):
        """
        设置图片的不透明度
        """
        if self.image_actor:
            self.image_actor.GetProperty().SetOpacity(opacity)

    def get_image_actor(self):
        """
        获取 vtkImageActor
        """
        return self.image_actor


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("DICOM Viewer")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 创建视图窗口
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        self.layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # 图片处理对象列表
        self.image_handlers = []
        self.current_image_index = 0  # 当前处理的图片索引

        # 创建菜单和按钮
        self.create_menu()
        self.create_switch_button()

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = file_menu.addAction("Open DICOM Files")
        open_action.triggered.connect(self.open_files)

    def create_switch_button(self):
        """
        创建切换图片的按钮
        """
        self.switch_button = QPushButton("Switch Image", self)
        self.switch_button.clicked.connect(self.switch_image)
        self.layout.addWidget(self.switch_button)

    def open_files(self):
        file_dialog = QFileDialog(self, "Select DICOM Files")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters(["DICOM Files (*.dcm)", "All Files (*)"])

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            # 只允许打开两个文件
            if len(selected_files) > 2:
                selected_files = selected_files[:2]

            for i, file in enumerate(selected_files):
                # 动态分配颜色
                color = (1, 0, 0) if len(self.image_handlers) == 0 else (0, 1, 0)
                vtk_image = self.load_dicom_file(file)
                if vtk_image:
                    image_handler = ImageHandler(vtk_image, color)
                    self.image_handlers.append(image_handler)

            # 如果已经加载了文件，则初始化视图
            if self.image_handlers:
                self.update_view()

    def load_dicom_file(self, filename):
        reader = itk.ImageFileReader[itk.Image[itk.SS, 3]].New()
        dicom_io = itk.GDCMImageIO.New()
        reader.SetImageIO(dicom_io)
        reader.SetFileName(filename)

        try:
            reader.Update()
        except Exception as e:
            print(f"Error reading DICOM file: {e}")
            return None

        itk_image = reader.GetOutput()
        vtk_image = self.itk_to_vtk_image(itk_image)
        return vtk_image

    def itk_to_vtk_image(self, itk_image):
        itk_array = itk.GetArrayViewFromImage(itk_image)
        vtk_image = vtk.vtkImageData()

        depth, height, width = itk_array.shape
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.AllocateScalars(vtk.VTK_SHORT, 1)

        vtk_data_array = numpy_to_vtk(itk_array.ravel(), deep=True, array_type=vtk.VTK_SHORT)
        vtk_image.GetPointData().SetScalars(vtk_data_array)

        return vtk_image

    def update_view(self):
        """
        更新视图，显示当前处理的图片
        """
        self.renderer.RemoveAllViewProps()  # 清除所有对象

        for i, handler in enumerate(self.image_handlers):
            actor = handler.get_image_actor()
            if i == self.current_image_index:
                actor.GetProperty().SetOpacity(1.0)  # 当前图片不透明
            else:
                actor.GetProperty().SetOpacity(0.5)  # 其他图片半透明
            self.renderer.AddActor(actor)

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def switch_image(self):
        """
        切换当前处理的图片
        """
        if len(self.image_handlers) > 1:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_handlers)
            self.update_view()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    app.shutdown()