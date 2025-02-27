import sys
import vtk
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
import itk

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

        self.style = None  # 初始化为 None

        # 创建菜单
        self.create_menu()

        self.renderer.SetUseDepthPeeling(True)  # 启用深度剥离
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.1)
        
    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = file_menu.addAction("Open DICOM Files")
        open_action.triggered.connect(self.open_files)
        
    def open_files(self):
        file_dialog = QFileDialog(self, "Select DICOM Files")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilters(["DICOM Files (*.dcm)", "All Files (*)"])

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if len(selected_files) >= 2:
                vtk_image_1 = self.load_dicom_file(selected_files[0])
                vtk_image_2 = self.load_dicom_file(selected_files[1])
                if vtk_image_1 and vtk_image_2:
                    self.visualize_vtk_images(vtk_image_1, vtk_image_2)

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

    def visualize_vtk_images(self, vtk_image_1, vtk_image_2):
        def create_image_actor(vtk_image, slice_orientation):
            reslice = vtk.vtkImageReslice()
            reslice.SetInputData(vtk_image)
            reslice.SetOutputDimensionality(2)
            
            if slice_orientation == 'axial':
                reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 1, 0, 0, 0, 1)
                reslice.SetResliceAxesOrigin(0, 0, vtk_image.GetDimensions()[2] // 2)
            elif slice_orientation == 'coronal':
                reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, 1, 0)
                reslice.SetResliceAxesOrigin(0, 768-316, 0)
            elif slice_orientation == 'sagittal':
                reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)
                reslice.SetResliceAxesOrigin(vtk_image.GetDimensions()[0] // 2, 0, 0)

            reslice.SetInterpolationModeToLinear()
            '''
            mapper = vtk.vtkImageMapper()
            mapper.SetInputConnection(reslice.GetOutputPort())
            mapper.SetColorWindow(255)
            mapper.SetColorLevel(127.5)
            '''
            actor = vtk.vtkImageActor()
            #actor.SetMapper(mapper)
            actor.GetMapper().SetInputConnection(reslice.GetOutputPort())
            return actor

        image_actor1 = create_image_actor(vtk_image_1, 'coronal')
        image_actor2 = create_image_actor(vtk_image_2, 'coronal')
        image_actor2.GetProperty().SetOpacity(0.5)
        image_actor1.GetProperty().SetColorWindow(600)
        image_actor2.GetProperty().SetColorWindow(600)

        self.renderer.AddActor(image_actor1)
        self.renderer.AddActor(image_actor2)

        self.renderer.ResetCamera()

        # 初始化 MouseInteractorStyle，并设置 image_actor2 为可拖动的对象
        self.style = MouseInteractorStyle(self.renderer, image_actor2)
        self.interactor.SetInteractorStyle(self.style)

        self.vtk_widget.GetRenderWindow().Render()
        self.interactor.Initialize()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
    app.shutdown()
