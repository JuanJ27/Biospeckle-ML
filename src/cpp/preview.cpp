// Configure OpenCV for ROOT
#pragma cling add_include_path("/usr/include/opencv4")
#pragma cling add_library_path("/usr/lib/x86_64-linux-gnu")
#pragma cling load("libopencv_core.so", "libopencv_imgproc.so", "libopencv_highgui.so", "libopencv_videoio.so")

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Función para calcular la matriz de co-ocurrencia
Mat compute_com(const Mat& Thsp, int n_levels = 256, int offset = 1) 
{
    Mat MCo = Mat::zeros(n_levels, n_levels, CV_32SC1); // Matriz de salida
    for (int r = 0; r < Thsp.rows; r++) {
        const uchar* row_ptr = Thsp.ptr<uchar>(r);
        for (int i = 0; i < Thsp.cols - offset; i++) {
            int i_val = row_ptr[i];
            int j_val = row_ptr[i + offset];
            MCo.at<int>(i_val, j_val)++; // Construye la matriz de co-ocurrencia
        }
    }
    return MCo;


}int preview() 
{
    // Abrir la cámara por defecto
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: No se pudo abrir la cámara." << endl;
        return -1;
    }

vector<Mat> frames; // Almacenar frames
const int max_frames = 300; // Número máximo de frames a procesar

while (true) 
{
    Mat frame;
    if (!cap.read(frame)) 
    {
        cout << "Error: No se pudo leer el fotograma." << endl;
        break;
    }

    // Convertir a escala de grises
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    frames.push_back(gray);

    // Limitar el número de frames almacenados
    if (frames.size() > max_frames) 
    {
        frames.erase(frames.begin(), frames.begin() + (frames.size() - max_frames));
    }

    // Procesar cuando tengamos suficientes frames
    if (frames.size() == max_frames) 
    {
        int indiceCol = frames[0].cols / 2; // Columna central
        int rows = frames[0].rows;
        Mat Thsp(rows, max_frames, CV_8UC1);

        // Extraer la columna central de cada frame
        for (int i = 0; i < max_frames; i++) {
            Mat col = frames[i].col(indiceCol);
            col.copyTo(Thsp.col(i));
        }

        // Calcular la matriz de co-ocurrencia
        Mat MCo = compute_com(Thsp);

        // Preparar visualización con normalización logarítmica
        Mat MCo_log;
        MCo.convertTo(MCo_log, CV_32FC1);
        MCo_log += 1; // Evitar log(0)
        log(MCo_log, MCo_log);
        Mat MCo_disp;
        normalize(MCo_log, MCo_disp, 0, 255, NORM_MINMAX, CV_8UC1);

        // Mostrar imágenes
        imshow("Thsp", Thsp);
        imshow("MCo", MCo_disp);
    }

    // Mostrar el frame original
    imshow("frame", frame);
    char key = (char)waitKey(1);
    if (key == 'q') { // Salir con la tecla 'q'
        break;
    }
}

// Liberar recursos
cap.release();
destroyAllWindows();
return 0;

}
