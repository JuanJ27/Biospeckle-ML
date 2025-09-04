// Configure OpenCV for ROOT
#pragma cling add_include_path("/usr/include/opencv4")
#pragma cling add_library_path("/usr/lib/x86_64-linux-gnu")
#pragma cling load("libopencv_core.so", "libopencv_imgproc.so", "libopencv_highgui.so", "libopencv_videoio.so")


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

using namespace cv;
using namespace std;

// Función para calcular la matriz de coocurrencia
Mat compute_com(const Mat& Thsp, int n_levels = 256, int offset = 1) {
    Mat MCo = Mat::zeros(n_levels, n_levels, CV_32SC1);
    for (int r = 0; r < Thsp.rows; r++) {
        const uchar* row_ptr = Thsp.ptr<uchar>(r);
        for (int c = 0; c < Thsp.cols - offset; c++) {
            int i_val = row_ptr[c];
            int j_val = row_ptr[c + offset];
            MCo.at<int>(i_val, j_val)++;
        }
    }
    return MCo;
}

// Función para calcular el momento de inercia de la matriz de co-ocurrencia
double computeMomentOfInertia(const Mat& MCo) {
    // Si MCo es de tipo int, lo convertimos a float para la suma
    Mat MCo_float;
    if (MCo.type() != CV_32F)
        MCo.convertTo(MCo_float, CV_32F);
    else
        MCo_float = MCo;
    
    double MI = 0.0;
    for (int i = 0; i < MCo_float.rows; i++) {
        for (int j = 0; j < MCo_float.cols; j++) {
            float val = MCo_float.at<float>(i, j);
            MI += val * ((i - j) * (i - j));
        }
    }
    return MI;
}

// Función para calcular la imagen GD (Generalised Differences)
// I0(i,j) = sum_{k=0}^{N-2} sum_{l=k+1}^{N-1} | I_k(i,j) - I_l(i,j) |
// Se recorre cada pixel y se suman las diferencias absolutas de cada par de frames.
Mat computeGD(const vector<Mat>& frames) {
    if (frames.empty())
        return Mat();
    
    int rows = frames[0].rows, cols = frames[0].cols;
    int N = frames.size();
    Mat GD = Mat::zeros(rows, cols, CV_32FC1);
    
    // Recorrer cada par de frames (solo una mitad, ya que es simétrica)
    for (int k = 0; k < N - 1; k++) {
        for (int l = k + 1; l < N; l++) {
            Mat diff;
            absdiff(frames[k], frames[l], diff);
            diff.convertTo(diff, CV_32FC1);
            GD += diff;
        }
    }
    // (Opcional) Normalizar GD para visualizar
    normalize(GD, GD, 0, 255, NORM_MINMAX, CV_32FC1);
    GD.convertTo(GD, CV_8UC1);
    return GD;
}

// Función para calcular la imagen de Fujii
// Para cada píxel se calcula: F(i,j) = promedio_{k}( |I_{k+1}(i,j)-I_k(i,j)| / (I_{k+1}(i,j)+I_k(i,j)+epsilon) )
Mat computeFujiiImage(const vector<Mat>& frames) {
    if (frames.size() < 2)
        return Mat();
    
    int rows = frames[0].rows, cols = frames[0].cols;
    int N = frames.size();
    Mat F = Mat::zeros(rows, cols, CV_32FC1);
    const float epsilon = 1e-6f;

    for (int k = 0; k < N - 1; k++) {
        Mat I1, I2;
        frames[k].convertTo(I1, CV_32FC1);
        frames[k+1].convertTo(I2, CV_32FC1);
        Mat diff, sumImg, ratio;
        absdiff(I2, I1, diff);
        add(I2, I1, sumImg);
        sumImg += epsilon;
        divide(diff, sumImg, ratio);
        F += ratio;
    }
    F /= (N - 1);
    normalize(F, F, 0, 255, NORM_MINMAX, CV_32FC1);
    F.convertTo(F, CV_8UC1);
    return F;
}

// funcion para etiquetar los resutlados
string get_tag(const string& path) {
    size_t last_slash = path.find_last_of('/');
    string filename = (last_slash == string::npos) ? path : path.substr(last_slash + 1);
    size_t last_dot = filename.find_last_of('.');
    string tag = (last_dot == string::npos) ? filename : filename.substr(0, last_dot);
    return tag;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <ruta_video>" << endl;
        return -1;
    }

    string video_path = argv[1];
    cout << "Cargando video: " << video_path << endl;

    // Extraer el tag
    string tag = get_tag(video_path);

    // Generar nombres de archivos
    string mi_file = tag + "_MI.txt";
    string thsp_yaml = tag + "_THSP.yaml";
    string mco_yaml = tag + "_MCo.yaml";
    // string mcolog_yaml = tag + "_MCoLog.yaml";
    string gd_bin = tag + "_GD.bin";
    string fujii_bin = tag + "_Fujii.bin";
    string thsp_img = tag + "_Thsp.png";
    //string mco_img = tag + "_MCo.png";
    string mcolog_img = tag + "_MCoLog.png";
    string gd_img = tag + "_GD.png";
    string fujii_img = tag + "_Fujii.png";
    // string gd_log_img = tag + "_GDLog.png";
    // string fujii_log_img = tag + "_FujiiLog.png";


    // Abrir el video almacenado
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cout << "Error: No se pudo abrir el video." << endl;
        return -1;
    }

    vector<Mat> frames;
    const int max_frames = 314;

    // Leer hasta max_frames o hasta que no queden más frames
    while ((int)frames.size() < max_frames) {
        Mat frame;
        if (!cap.read(frame)) {
            break; // Se terminó el video
        }

        // Convertir el frame a escala de grises
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        frames.push_back(gray);
    }

    // Comprobar si se leyeron frames
    if (frames.empty()) {
        cout << "Error: No se pudieron cargar frames." << endl;
        return -1;
    }

    // Ajustar max_frames si se leyeron menos frames de lo esperado
    int total_frames = frames.size();

    // Preparar la imagen THSP
    int indiceCol = frames[0].cols / 2;
    int rows = frames[0].rows;
    Mat Thsp(rows, total_frames, CV_8UC1);

    // Extraer la columna central de cada frame
    for (int i = 0; i < total_frames; i++) {
        Mat col = frames[i].col(indiceCol);
        col.copyTo(Thsp.col(i));
    }

    // Calcular la matriz de coocurrencia
    Mat MCo = compute_com(Thsp);
    // Calcular momento de inercia (MI) de la matriz de co-ocurrencia
    double MI = computeMomentOfInertia(MCo);
    cout << "Momento de Inercia (MI): " << MI << endl;

    // Calcular la imagen GD a partir de los frames leídos y aplicar colormap JET
    Mat GD_gray = computeGD(frames);
    Mat GD;
    applyColorMap(GD_gray, GD, COLORMAP_JET);

    // // Versión en escala logarítmica de GD
    // Mat GD_gray_log;
    // GD_gray.convertTo(GD_gray_log, CV_32F);
    // GD_gray_log += 1;
    // log(GD_gray_log, GD_gray_log);
    // normalize(GD_gray_log, GD_gray_log, 0, 255, NORM_MINMAX);
    // GD_gray_log.convertTo(GD_gray_log, CV_8UC1);
    // Mat GD_log;
    // applyColorMap(GD_gray_log, GD_log, COLORMAP_JET);

    // Calcular la imagen de Fujii y aplicar colormap JET
    Mat fujiiImg_gray = computeFujiiImage(frames);
    Mat fujiiImg;
    applyColorMap(fujiiImg_gray, fujiiImg, COLORMAP_JET);

    // // Versión en escala logarítmica de Fujii
    // Mat fujiiImg_gray_log;
    // fujiiImg_gray.convertTo(fujiiImg_gray_log, CV_32F);
    // fujiiImg_gray_log += 1;
    // log(fujiiImg_gray_log, fujiiImg_gray_log);
    // normalize(fujiiImg_gray_log, fujiiImg_gray_log, 0, 255, NORM_MINMAX);
    // fujiiImg_gray_log.convertTo(fujiiImg_gray_log, CV_8UC1);
    // Mat fujiiImg_log;
    // applyColorMap(fujiiImg_gray_log, fujiiImg_log, COLORMAP_JET);

    // Preparar visualización con normalización logarítmica
    Mat MCo_log;
    MCo.convertTo(MCo_log, CV_32FC1);
    MCo_log += 1;
    log(MCo_log, MCo_log);

    Mat MCo_disp;
    normalize(MCo_log, MCo_disp, 0, 255, NORM_MINMAX, CV_8UC1);
    Mat MCo_normal_disp;
    normalize(MCo, MCo_normal_disp, 0, 255, NORM_MINMAX, CV_8UC1);

    // Guardar MI
    ofstream mi_out(mi_file);
    if (mi_out.is_open()) {
        mi_out << "Momento de Inercia (MI): " << MI << endl;
        mi_out.close();
    } else {
        cout << "Error al abrir " << mi_file << endl;
    }

    // Guardar THSP
    FileStorage fs_thsp(thsp_yaml, FileStorage::WRITE);
    if (fs_thsp.isOpened()) {
        fs_thsp << "THSP" << Thsp;
        fs_thsp.release();
    } else {
        cout << "Error al abrir " << thsp_yaml << endl;
    }

    // Guardar MCo
    FileStorage fs_mco(mco_yaml, FileStorage::WRITE);
    if (fs_mco.isOpened()) {
        fs_mco << "MCo" << MCo;
        fs_mco.release();
    } else {
        cout << "Error al abrir " << mco_yaml << endl;
    }

    // // Guardar MColog
    // FileStorage fs_mcolog(mcolog_yaml, FileStorage::WRITE);
    // if (fs_mcolog.isOpened()) {
    //     fs_mcolog << "MCo" << MCo_log;
    //     fs_mcolog.release();
    // } else {
    //     cout << "Error al abrir " << mcolog_yaml << endl;
    // }

    // Guardar GD
    ofstream gd_out(gd_bin, ios::binary);
    if (gd_out.is_open()) {
        gd_out.write(reinterpret_cast<const char*>(GD_gray.data), GD_gray.total() * GD_gray.elemSize());
        gd_out.close();
    } else {
        cout << "Error al abrir " << gd_bin << endl;
    }

    // Guardar Fujii
    ofstream fujii_out(fujii_bin, ios::binary);
    if (fujii_out.is_open()) {
        fujii_out.write(reinterpret_cast<const char*>(fujiiImg_gray.data), fujiiImg_gray.total() * fujiiImg_gray.elemSize());
        fujii_out.close();
    } else {
        cout << "Error al abrir " << fujii_bin << endl;
    }

    // Guardar imágenes
    imwrite(thsp_img, Thsp);
    cout << "Imagen THSP guardada en " << thsp_img << endl;
    // imwrite(mco_img, MCo_normal_disp);
    // cout << "Imagen MCo guardada en " << mco_img << endl;
    imwrite(gd_img, GD);
    cout << "Imagen GD guardada en " << gd_img << endl;
    imwrite(fujii_img, fujiiImg);
    cout << "Imagen Fujii guardada en " << fujii_img << endl;
    imwrite(mcolog_img, MCo_disp);
    cout << "Imagen MCoLog guardada en " << mcolog_img << endl;
    // imwrite(gd_log_img, GD_log);
    // cout << "Imagen GDLog guardada en " << gd_log_img << endl;
    // imwrite(fujii_log_img, fujiiImg_log);
    // cout << "Imagen FujiiLog guardada en " << fujii_log_img << endl;    
    

    cout << "Archivos generados con éxito para el video: " << video_path << endl;

    // Mostrar la THSP y la matriz hasta que se presione 'q'
    while (true) {
        imshow("THSP", Thsp);
        imshow("MCo Log", MCo_disp);
        // imshow("MCo", MCo_normal_disp);
        imshow("GD", GD);
        imshow("Fujii", fujiiImg);
        // imshow("GD Log", GD_log);
        // imshow("Fujii Log", fujiiImg_log);
        char key = (char)waitKey(30);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}