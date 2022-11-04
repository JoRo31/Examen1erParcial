
/*Examen Practico 
*Romero Iglesias Jorge
* 5BV1
*Vision Artificial
* 04/11/2022
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#define pi 3.1416
#define e 2.72


using namespace cv;
using namespace std;


vector<vector<float>> generateKernel(int kSize, int sigma) {
	int amountSlide = (kSize - 1) / 2;
	vector<vector<float>> v(kSize, vector<float>(kSize, 0));
	// si el centro es (0,0)
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float resultado = (1 / (2 * pi * sigma * sigma)) * pow(e, -((i * i + j * j) / (2 * sigma * sigma)));
			v[i + amountSlide][j + amountSlide] = resultado;
			cout << "El valor del kernel es: " << resultado << endl;
		}
	}
	return v;
}

float applyFilterToPix(Mat original, vector<vector<float>> kernel, int kSize, int x, int y) {
	int rows = original.rows;
	int cols = original.cols;
	int amountSlide = (kSize - 1) / 2;
	float sumFilter = 0;
	float sumKernel = 0;
	for (int i = -amountSlide; i <= amountSlide; i++)
	{
		for (int j = -amountSlide; j <= amountSlide; j++)
		{
			float kTmp = kernel[i + amountSlide][j + amountSlide];
			int tmpX = x + i;
			int tmpY = y + j;
			float tmp = 0;
			if (!(tmpX < 0 || tmpX >= cols || tmpY < 0 || tmpY >= rows)) {
				tmp = original.at<uchar>(Point(tmpX, tmpY));
			}

			sumFilter += (kTmp * tmp);
			sumKernel += kTmp;
		}
	}
	return sumFilter / sumKernel;
}

Mat applyFilterToMat(Mat original, vector<vector<float>> kernel, int kSize) {
	Mat filteredImg(original.rows, original.cols, CV_8UC1);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++) {
			filteredImg.at<uchar>(Point(i, j)) = uchar(applyFilterToPix(original, kernel, kSize, i, j));
		}
	}
	return filteredImg;
}

/////////////////////////Inicio de la funcion principal///////////////////
int main()
{
	int sigma = 1;
	int kSize = 3;
	cout << "Ingresa el tamano del kernel" << endl;

	cin >> kSize;

	if (kSize % 2 == 0) {
		cout << "Valor de kernel invalido" << endl;
		exit(0);
	}

	cout << "Ingresa sigma" << endl;

	cin >> sigma;

	/********Declaracion de variables generales*********/
	char NombreImagen[] = "Lenna.png";
	Mat imagen,Image_ecualiz;; // Matriz que contiene nuestra imagen sin importar el formato
	/************************/

	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	/************************/

	/************Procesos*********/
	int fila_original = imagen.rows;
	int columna_original = imagen.cols;//Lectura de cuantas columnas

	cout << "filas de la imagen original: " << fila_original << endl;
	cout << "columnas de la imagen original: " << columna_original << endl;



	Mat imagenGrisesNTSC(fila_original, columna_original, CV_8UC1);

	// Pasamos a escala de grises
	for (int i = 0; i < fila_original; i++)
	{
		for (int j = 0; j < columna_original; j++)
		{
			double azul = imagen.at<Vec3b>(Point(j, i)).val[0];  // B
			double verde = imagen.at<Vec3b>(Point(j, i)).val[1]; // G
			double rojo = imagen.at<Vec3b>(Point(j, i)).val[2];  // R

			imagenGrisesNTSC.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}

	vector<vector<float>> kernel = generateKernel(kSize, sigma);
	Mat filtrada = applyFilterToMat(imagenGrisesNTSC, kernel, kSize);

	/********Histogramas*********/

	//Variables para el histograma
	int histSize = 256;
	/// el rango del nivel del gris 0-255
	float range[] = { 0, 256 };
	const float* histRange = { range };

	/// imagen del histograma
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat ecualizHistImag(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	//calcular el histograma
	Mat original_hist, hist_nomal, ehist_ecualiz, ehist_ecualiz_nomal;
	calcHist(&imagenGrisesNTSC, 1, 0, Mat(), original_hist, 1, &histSize, &histRange, true, false);

	/// Normalizar el resultado a [ 0, histImage.rows ]
	normalize(original_hist, hist_nomal, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//Ecualizacion del histograma a partir de una imagen en escala de grises	
	equalizeHist(imagenGrisesNTSC,Image_ecualiz);
	calcHist(&Image_ecualiz, 1, 0, Mat(), ehist_ecualiz, 1, &histSize, &histRange, true, false);


	//Normalizar el histograma ecualizado
	normalize(ehist_ecualiz, ehist_ecualiz_nomal, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// dibujar los histogramas
	for (int i = 1; i < histSize; i++)
	{
		//Línea de ancho 2 (bin_w = 512 ancho / 256 valores de escala de grises)
		line(histImage,
			Point(bin_w * (i), hist_w),
			Point(bin_w * (i), hist_h - cvRound(hist_nomal.at<float>(i))),
			Scalar(255, 0, 0), bin_w, 8, 0);

		line(ecualizHistImag,
			Point(bin_w * (i), hist_w),
			Point(bin_w * (i), hist_h - cvRound(ehist_ecualiz_nomal.at<float>(i))),
			Scalar(0, 255, 0), bin_w, 8, 0);
	}

	//**************** |G| **********************

	int cols = imagenGrisesNTSC.cols;
	int rows = imagenGrisesNTSC.rows;

	cout << "filas de la imagen original: " << fila_original << endl;
	cout << "columnas de la imagen original: " << columna_original << endl;

	cout << "filas de la imagen en escala de gris: " << rows << endl;
	cout << "columnas de la imagen en escala de gris: " << cols << endl;

	//Creacion del operador sobel en la dirección x
	int sobel_x[3][3] = { -1,0,1,-2,0,2,-1,0,1 };
	//Creacion del operador sobel en la dirección y
	int sobel_y[3][3] = { -1,-2,-1,0,0,0,1,2,1 };

	int radius = 1;

	//Manejo de los bordes 
	Mat src;
	copyMakeBorder(imagenGrisesNTSC, src, radius, radius, radius, radius, BORDER_REFLECT101);

	//Creacion de las gradiantes
	Mat gradient_x = imagenGrisesNTSC.clone();
	Mat gradient_y = imagenGrisesNTSC.clone();
	Mat gradient_f = imagenGrisesNTSC.clone();

	int max = 0;

	//Iteracion en la imagen 
	for (int r = radius; r < src.rows - radius; ++r)
	{
		for (int c = radius; c < src.cols - radius; ++c)
		{
			int s = 0;

			//Iteracion en el kernel
			for (int i = -radius; i <= radius; ++i)
			{
				for (int j = -radius; j <= radius; ++j)
				{
					s += src.at<uchar>(r + i, c + j) * sobel_x[i + radius][j + radius];
				}
			}
			gradient_x.at<uchar>(r - radius, c - radius) = s / 30;
		}
	}

	Mat absGrad_x;
	convertScaleAbs(gradient_x, absGrad_x);
 

	//Iteracion en la imagen
	for (int r = radius; r < src.rows - radius; ++r)
	{
		for (int c = radius; c < src.cols - radius; ++c)
		{
			int s = 0;

			// Iteracion en el kernel
			for (int i = -radius; i <= radius; ++i)
			{
				for (int j = -radius; j <= radius; ++j)
				{
					s += src.at<uchar>(r + i, c + j) * sobel_y[i + radius][j + radius];
				}
			}

			gradient_y.at<uchar>(r - radius, c - radius) = s / 30;

		}
	}

	Mat absGrad_y;
	convertScaleAbs(gradient_y, absGrad_y);

	//Calculo de la magnitud del gradiente
	for (int i = 0; i < gradient_f.rows; i++)
	{
		for (int j = 0; j < gradient_f.cols; j++)
		{
			gradient_f.at<uchar>(i, j) = sqrt(pow(gradient_x.at<uchar>(i, j), 2) + pow(gradient_y.at<uchar>(i, j), 2));

			if (gradient_f.at<uchar>(i, j) > 240)
				gradient_f.at<uchar>(i, j) = 100;
			else
				gradient_f.at<uchar>(i, j) = 0;
		}
	}

	int rows_filt = filtrada.rows;
	int cols_filt = filtrada.cols;

	int rows_ecua = Image_ecualiz.rows;
	int cols_ecua = Image_ecualiz.cols;

	int rows_ab = gradient_f.rows;
	int cols_ab = gradient_f.cols;

	cout << "filas de la imagen en suavizada: " << rows_filt << endl;
	cout << "columnas de la imagen suavizada: " << cols_filt << endl;

	cout << "filas de la imagen ecualizada: " << rows_ecua << endl;
	cout << "columnas de la imagen ecualizada: " << cols_ecua << endl;

	cout << "filas de la imagen ecualizada: " << rows_ab << endl;
	cout << "columnas de la imagen ecualizada: " << cols_ab << endl;


	namedWindow("Imagen normal", WINDOW_AUTOSIZE);//Creaci de una ventana
	imshow("Imagen normal", imagen);

	namedWindow("Escala de gris", WINDOW_AUTOSIZE);//Creaci de una ventana
	imshow("Escala de gris", imagenGrisesNTSC);

	namedWindow("Imagen suavizada", WINDOW_AUTOSIZE);//Creaci de una ventana
	imshow("Imagen suavizada", filtrada);

	namedWindow("Imagen ecualizada", WINDOW_AUTOSIZE);//Creaci de una ventana
	imshow("Imagen ecualizada",Image_ecualiz);

	namedWindow("Histograma ecualizado", WINDOW_AUTOSIZE);//Creaci de una ventana
	imshow("Histograma ecualizado", ecualizHistImag);

	namedWindow("Magnitud del gradiante", WINDOW_AUTOSIZE);//Creaci de una ventana
	imshow("Magnitud del gradiante", gradient_f);


	waitKey(0); //Funcion para esperar
	return 1;
}
