import sys
import vtk
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog, QPushButton, QLineEdit, QLabel
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
import itk


class ImageHandler:
    def __init__(self, vtk_image):
        self.vtk_image = vtk_image
        self.image_actor = None
        self.slice_orientation = 'coronal'  # 默认视图方向
        self.slice_position = self.vtk_image.GetDimensions()[1] // 2  # 初始切片位置
        self.create_image_actor()

    def create_image_actor(self):
        """
        创建 vtkImageActor 并绑定到当前图片
        """
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self.vtk_image)
        reslice.SetOutputDimensionality(2)

        # 设置切片方向
        if self.slice_orientation == 'coronal':
            reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, 1, 0)
            reslice.SetResliceAxesOrigin(0, self.slice_position, 0)  # 仅沿 Y 轴移动切片

        reslice.SetInterpolationModeToLinear()

        # 直接使用灰度图像，不进行颜色映射
        self.image_actor = vtk.vtkImageActor()
        self.image_actor.GetMapper().SetInputConnection(reslice.GetOutputPort())

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

    def update_slice_position(self, position):
        """
        更新切片位置并重新渲染
        """
        self.slice_position = position
        self.create_image_actor()  # 重新创建图像 Actor


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

        # 禁用默认的交互方式（旋转、缩放等）
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleImage())  # 使用 Image 交互方式，禁用旋转

        # 图片处理对象列表
        self.image_handlers = []
        self.current_image_index = 0  # 当前处理的图片索引

        # 创建菜单和按钮
        self.create_menu()
        self.create_switch_button()
        self.create_slice_position_input()

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

    def create_slice_position_input(self):
        """
        创建切片位置输入框
        """
        self.slice_position_label = QLabel("Slice Position:", self)
        self.layout.addWidget(self.slice_position_label)

        self.slice_position_input = QLineEdit(self)
        self.slice_position_input.setPlaceholderText("Enter slice position")
        self.slice_position_input.returnPressed.connect(self.update_slice_position_from_input)
        self.layout.addWidget(self.slice_position_input)

    def open_files(self):
        file_dialog = QFileDialog(self, "Select DICOM Files")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters(["DICOM Files (*.dcm)", "All Files (*)"])

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            # 只允许打开两个文件
            if len(selected_files) > 2:
                selected_files = selected_files[:2]

            for file in selected_files:
                vtk_image = self.load_dicom_file(file)
                if vtk_image:
                    image_handler = ImageHandler(vtk_image)
                    self.image_handlers.append(image_handler)

            # 如果已经加载了文件，则初始化视图
            if self.image_handlers:
                self.update_view()
                self.update_slice_position_input()  # 初始化输入框的值

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
            self.update_slice_position_input()  # 更新输入框的值

    def update_slice_position_input(self):
        """
        更新输入框的值为当前图像的切片位置
        """
        if self.image_handlers:
            current_handler = self.image_handlers[self.current_image_index]
            self.slice_position_input.setText(str(current_handler.slice_position))

    def update_slice_position_from_input(self):
        """
        从输入框更新当前图像的切片位置
        """
        if self.image_handlers:
            try:
                position = int(self.slice_position_input.text())
                current_handler = self.image_handlers[self.current_image_index]
                current_handler.update_slice_position(position)
                self.update_view()  # 重新渲染视图
            except ValueError:
                print("Invalid input. Please enter an integer.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    app.shutdown()