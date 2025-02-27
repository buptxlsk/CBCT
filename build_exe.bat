@echo off
echo Activating the virtual environment...
call D:\DProgram_Files\Anaconda3\Scripts\activate.bat D:\DProgram_Files\Anaconda3\envs\CBCT

echo Building the application using PyInstaller...
pyinstaller -F --windowed --add-data "D:\DProgram_Files\Anaconda3\envs\CBCT\Lib\site-packages\vtk.libs;vtk.libs" ^
            --add-data "D:\DProgram_Files\Anaconda3\envs\CBCT\Lib\site-packages\itk_core.libs;itk_core.libs" ^
            --add-data "D:\DProgram_Files\Anaconda3\envs\CBCT\Lib\site-packages\itk;itk" ^
            --add-data "D:\DProgram_Files\Anaconda3\envs\CBCT\Lib\site-packages\pandas.libs;pandas.libs" ^
            --add-data "D:\DProgram_Files\Anaconda3\envs\CBCT\Lib\site-packages\pandas;pandas" ^
            --hidden-import pydicom.encoders.gdcm ^
            --hidden-import pydicom.encoders.pylibjpeg ^
            --hidden-import vtkmodules.util.data_model ^
            --hidden-import vtkmodules.all ^
            --collect-all vtkmodules test.py

echo Build complete!
pause
