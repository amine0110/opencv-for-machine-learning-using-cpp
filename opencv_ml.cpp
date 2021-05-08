
// EssaiOpencvDlg.cpp : fichier d'implémentation
//

#include "stdafx.h"
#include "EssaiOpencv.h"
#include "EssaiOpencvDlg.h"
#include "afxdialogex.h"



#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;


// boîte de dialogue CAboutDlg utilisée pour la boîte de dialogue 'À propos de' pour votre application

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Données de boîte de dialogue
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // Prise en charge de DDX/DDV

// Implémentation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// boîte de dialogue CEssaiOpencvDlg



CEssaiOpencvDlg::CEssaiOpencvDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_ESSAIOPENCV_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CEssaiOpencvDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CEssaiOpencvDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_FILTRE, &CEssaiOpencvDlg::OnBnClickedButtonFiltre)
	ON_BN_CLICKED(IDC_BUTTON_TEST, &CEssaiOpencvDlg::OnBnClickedButtonTest)
	ON_BN_CLICKED(IDC_BUTTON4, &CEssaiOpencvDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CEssaiOpencvDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CEssaiOpencvDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CEssaiOpencvDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CEssaiOpencvDlg::OnBnClickedButton8)
	ON_EN_CHANGE(IDC_EDIT1, &CEssaiOpencvDlg::OnEnChangeEdit1)
	ON_BN_CLICKED(IDC_BUTTON2, &CEssaiOpencvDlg::OnBnClickedButton2)
END_MESSAGE_MAP()


// gestionnaires de messages pour CEssaiOpencvDlg

BOOL CEssaiOpencvDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Ajouter l'élément de menu "À propos de..." au menu Système.

	// IDM_ABOUTBOX doit se trouver dans la plage des commandes système.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Définir l'icône de cette boîte de dialogue.  L'infrastructure effectue cela automatiquement
	//  lorsque la fenêtre principale de l'application n'est pas une boîte de dialogue
	SetIcon(m_hIcon, TRUE);			// Définir une grande icône
	SetIcon(m_hIcon, FALSE);		// Définir une petite icône

	// TODO: ajoutez ici une initialisation supplémentaire

	return TRUE;  // retourne TRUE, sauf si vous avez défini le focus sur un contrôle
}

void CEssaiOpencvDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// Si vous ajoutez un bouton Réduire à votre boîte de dialogue, vous devez utiliser le code ci-dessous
//  pour dessiner l'icône.  Pour les applications MFC utilisant le modèle Document/Vue,
//  cela est fait automatiquement par l'infrastructure.

void CEssaiOpencvDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // contexte de périphérique pour la peinture

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Centrer l'icône dans le rectangle client
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Dessiner l'icône
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// Le système appelle cette fonction pour obtenir le curseur à afficher lorsque l'utilisateur fait glisser
//  la fenêtre réduite.
HCURSOR CEssaiOpencvDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CEssaiOpencvDlg::OnBnClickedButtonFiltre()
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return;

	Mat edges;
	namedWindow("edges", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
		imshow("edges", edges);
		if (waitKey(0x1B) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
}



void CEssaiOpencvDlg::OnBnClickedButtonTest()
{
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return;
	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		medianBlur(gray, gray, 5);


		vector<Vec3f> circles;
		HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1,
			gray.rows / 16,  // change this value to detect circles with different distances to each other
			100, 30, 1, 30 // change the last two parameters
	   // (min_radius & max_radius) to detect larger circles
		);

		for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3i c = circles[i];
			Point center = Point(c[0], c[1]);
			// circle center
			circle(frame, center, 1, Scalar(0, 100, 100), 3,  CV_AA);
			// circle outline
			int radius = c[2];
			circle(frame, center, radius, Scalar(255, 0, 255), 3,  CV_AA);
		}

		imshow("pic", frame);

		

		if (waitKey(0x1B) >= 0) break;
	}

}





void CEssaiOpencvDlg::OnBnClickedButton4() //Knn
{
	const int K = 7;
	int li, col, k, accuracy;
	int train_sample_count = 100;
	int test_sample_count = 200;

	Mat trainData1(train_sample_count, 2, CV_32FC1);
	Mat trainData2(train_sample_count, 2, CV_32FC1);

	Mat testData1(test_sample_count, 2, CV_32FC1);
	Mat testData2(test_sample_count, 2, CV_32FC1);

	Mat testClasses1 = Mat::zeros(test_sample_count, 1, CV_32FC1);
	Mat testClasses2 = Mat::ones(test_sample_count, 1, CV_32FC1);
	Mat TrainClasses1 = Mat::zeros(train_sample_count, 1, CV_32FC1);
	Mat TrainClasses2 = Mat::ones(train_sample_count, 1, CV_32FC1);

	Mat trainDataGlobal;
	Mat TrainClasseG;
	Mat testDataGlobal;


	Mat Img = Mat::zeros(500, 500, CV_8UC3);
	Mat Img2 = Mat::zeros(500, 500, CV_8UC3);

	Mat Echantillon(1, 2, CV_32FC1); 
	Mat Echantillon1(1, 2, CV_32FC1);
	Mat Echantillon2(1, 2, CV_32FC1);
	RNG rng(12345);

	// Pour générer les données aléatoires
	rng.fill(trainData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80)); 
	rng.fill(trainData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80)); 
	rng.fill(trainData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80)); 
	rng.fill(trainData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80)); 
	rng.fill(testData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80));
	rng.fill(testData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80));
	rng.fill(testData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80));
	rng.fill(testData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80));
	vconcat(trainData1, trainData2, trainDataGlobal);
	vconcat(TrainClasses1, TrainClasses2, TrainClasseG);
	vconcat(testData1, testData2, testDataGlobal);

	Mat PlusProche(1, K, CV_32FC1);
	Mat Reponse, Dist;

	// déclaration du modele et lancement d'appretissage
	CvKNearest knn(trainDataGlobal, TrainClasseG, Reponse, false, K);
	
	// Pour tracer la frontière (on mettant les pixels de chaque classes en une couleur dans l'image)
	for (li = 0; li < Img.rows; li++)
	{
		for (col = 0; col < Img.cols; col++)
		{
			Echantillon.at<float>(0) = (float)(col);
			Echantillon.at<float>(1) = (float)(li);

			float rr = knn.find_nearest(Echantillon, K, Reponse, PlusProche, Dist);

			if (rr == 1)
			{
				Img.at < Vec3b>(li, col) = Vec3b(180, 0, 0); // BGR
			}
			else
			{
				Img.at < Vec3b>(li, col) = Vec3b(0, 180, 0);
			}
		}
	}

	// Pour placer les données d'apprentissage
	for (int i = 0; i < train_sample_count; i++) {
		CvPoint pt;
		pt.x = trainData1.at<float>(i, 0);
		pt.y = trainData1.at<float>(i, 1);
		circle(Img, pt, 2, CV_RGB(255, 0, 0), CV_FILLED); //RGB
		pt.x = trainData2.at<float>(i, 0);
		pt.y = trainData2.at<float>(i, 1);
		circle(Img, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);
	}
	imshow("sortie", Img); //cvWaitKey(0);
	

	// Partie test
	int VP = 0, FP = 0, VN = 0, FN = 0;


	for (int i = 0; i < test_sample_count; i++)
	{
		Echantillon1.at<float>(0) = (testData1.at<float>(i, 0));
		Echantillon1.at<float>(1) = (testData1.at<float>(i, 1));
		Echantillon2.at<float>(0) = (testData2.at<float>(i, 0));
		Echantillon2.at<float>(1) = (testData2.at<float>(i, 1));

		float rr1 = knn.find_nearest(Echantillon1, K, Reponse, PlusProche, Dist);
		float rr2 = knn.find_nearest(Echantillon2, K, Reponse, PlusProche, Dist);
		
		if (rr1 == 0)
			VP++;
		else
			FP++;

		if (rr2 == 0)
			FN++;
		else
			VN++;
	}

	for (li = 0; li < Img2.rows; li++)
	{
		for (col = 0; col < Img2.cols; col++)
		{
			Echantillon1.at<float>(0) = (float)(col);
			Echantillon1.at<float>(1) = (float)(li);
			Echantillon2.at<float>(0) = (float)(col);
			Echantillon2.at<float>(1) = (float)(li);
			float rr1 = knn.find_nearest(Echantillon1, K, Reponse, PlusProche, Dist);
			float rr2 = knn.find_nearest(Echantillon2, K, Reponse, PlusProche, Dist);




			if (rr1 == 1)
			{
				Img2.at < Vec3b>(li, col) = Vec3b(180, 0, 0); // BGR
			}
			else
			{
				Img2.at < Vec3b>(li, col) = Vec3b(0, 180, 0);
			}
		}
	}

	for (int i = 0; i < test_sample_count; i++) {
		CvPoint pt;
		pt.x = testData1.at<float>(i, 0);
		pt.y = testData1.at<float>(i, 1);
		circle(Img2, pt, 2, CV_RGB(255, 0, 0), CV_FILLED); //RGB
		pt.x = testData2.at<float>(i, 0);
		pt.y = testData2.at<float>(i, 1);
		circle(Img2, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);
	}

	imshow("test", Img2); cvWaitKey(0);
	
	
	
}


void CEssaiOpencvDlg::OnBnClickedButton5() // SVM des points
{
	int li = 0;
	int	col = 0;
	int	k, accuracy;
	// déclaration des dimensions des tableaux
	int train_sample_count = 100;
	int test_sample_count = 200;


	// déclaration des tableaux de données d'apprentissage
	Mat trainData1(train_sample_count, 2, CV_32FC1);
	Mat trainData2(train_sample_count, 2, CV_32FC1);


	// déclaration des tableaux données de tests 
	Mat testData1(test_sample_count, 2, CV_32FC1);
	Mat testData2(test_sample_count, 2, CV_32FC1);


	// déclaration des deux classes pour l'apprentissage et pour les tests : une classe aura 0 et l'autre aura 1
	Mat testClasses1 = Mat::zeros(test_sample_count, 1, CV_32FC1);
	Mat testClasses2 = Mat::ones(test_sample_count, 1, CV_32FC1);
	Mat TrainClasses1 = Mat::zeros(train_sample_count, 1, CV_32FC1);
	Mat TrainClasses2 = Mat::ones(train_sample_count, 1, CV_32FC1);

	// déclaration des matrices pour le regroupement des données et des classes
	Mat trainDataGlobal;
	Mat TrainClasseG;
	Mat testDataGlobal;


	// création d'une image noire
	Mat Img = Mat::zeros(500, 500, CV_8UC3);
	Mat Img2 = Mat::zeros(500, 500, CV_8UC3);

	// Déclaration des échantillons avec deux colones ( x et y )
	Mat Echantillon(1, 2, CV_32FC1);
	Mat Echantillon1(1, 2, CV_32FC1);
	Mat Echantillon2(1, 2, CV_32FC1);

	// Random number generator pour générer des numéros alétoires
	RNG rng(12345);

	// Remplir les données de test et d'apprentissage avec la loi normale
	rng.fill(trainData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80)); 
	rng.fill(trainData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80)); 
	rng.fill(trainData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80));
	rng.fill(trainData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80));
	rng.fill(testData1.col(0), RNG::NORMAL, Scalar(50), Scalar(80));
	rng.fill(testData1.col(1), RNG::NORMAL, Scalar(150), Scalar(80));
	rng.fill(testData2.col(0), RNG::NORMAL, Scalar(300), Scalar(80));
	rng.fill(testData2.col(1), RNG::NORMAL, Scalar(190), Scalar(80));

	// concaténation des matrices
	vconcat(trainData1, trainData2, trainDataGlobal);
	vconcat(TrainClasses1, TrainClasses2, TrainClasseG);
	vconcat(testData1, testData2, testDataGlobal);


	// déclaration des paramètres du SVM
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 1.0; 
	params.term_crit = cvTermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);
	params.C = 0.1; 

	// déclaration du classifieur SVM
	CvSVM SVM;
	SVM.train_auto(trainDataGlobal, TrainClasseG, Mat(), Mat(), params); 
	int VP=0, FP=0, VN=0, FN=0;

	for (li = 0; li < Img.rows; li++)
	{
		for (col = 0; col < Img.cols; col++)
		{
			Echantillon.at<float>(0) = (float)(col);
			Echantillon.at<float>(1) = (float)(li);
			float rr = SVM.predict(Echantillon);
			if (rr == 0)
			{
				Img.at<Vec3b>(li, col) = Vec3b(0, 0, 0);
			
			}
			else
			{
				Img.at<Vec3b>(li, col) = Vec3b(255, 255, 255);
				
			}
		}
	}


	for (int i = 0; i < train_sample_count; i++)
	{
		CvPoint pt;
		pt.x = trainData1.at<float>(i, 0);
		pt.y = trainData1.at<float>(i, 1);
		circle(Img, pt, 2, CV_RGB(255, 0, 0), CV_FILLED);
		pt.x = trainData2.at<float>(i, 0);
		pt.y = trainData2.at<float>(i, 1);
		circle(Img, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);
	}
	imshow("sortie", Img);

	for (int i = 0; i < test_sample_count; i++)
	{
		Echantillon1.at<float>(0) = (testData1.at<float>(i, 0));
		Echantillon1.at<float>(1) = (testData1.at<float>(i, 1));
		Echantillon2.at<float>(0) = (testData2.at<float>(i, 0));
		Echantillon2.at<float>(1) = (testData2.at<float>(i, 1));


		float pp1 = SVM.predict(Echantillon1);
		float pp2 = SVM.predict(Echantillon2);
		if (pp1 == 0)
			VP++;
		else
			FP++;

		if (pp2 == 0)
			FN++;
		else
			VN++;
		

	}

	// Cette partie, pour tracer la frontière
	for (li = 0; li < Img2.rows; li++)
	{
		for (col = 0; col < Img2.cols; col++)
		{
			Echantillon1.at<float>(0) = (float)(col);
			Echantillon1.at<float>(1) = (float)(li);
			float rr1 = SVM.predict(Echantillon1);
			if (rr1 == 0)
			{
				Img2.at<Vec3b>(li, col) = Vec3b(0, 0, 0);
				
			}
			else
			{
				Img2.at<Vec3b>(li, col) = Vec3b(255, 255, 255);
				
			}
		}
	}
	// Cette partie pour représenter les données de test
	for (int i = 0; i < test_sample_count; i++)
	{
		CvPoint pt;
		pt.x = testData1.at<float>(i, 0);
		pt.y = testData1.at<float>(i, 1);
		circle(Img2, pt, 2, CV_RGB(255, 0, 0), CV_FILLED);
		pt.x = testData2.at<float>(i, 0);
		pt.y = testData2.at<float>(i, 1);
		circle(Img2, pt, 2, CV_RGB(0, 255, 0), CV_FILLED);
	}

	for (int i = 0; i < test_sample_count; i++)
	{

	}

	imshow("test", Img2);
	cvWaitKey(0);

}


void CEssaiOpencvDlg::OnBnClickedButton6() // dico
{
	char* filename = new char[100];
	Mat input;
	vector<KeyPoint> keypoints;
	Mat descriptor;
	Mat featuresUnclustered;
	SiftDescriptorExtractor detector;

	string Nom[4] = { "accordion","airplanes","anchor","ant" };
	for (int i = 0; i < 4; i++)
	{
		for (int f = 1; f <= 10; f++)
		{
			sprintf(filename, "\\image_%04d.jpg", f);
			string nomf = ".\\" + Nom[i] + filename;
			input = imread(nomf.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			detector.detect(input, keypoints);
			detector.compute(input, keypoints, descriptor);
			featuresUnclustered.push_back(descriptor);
			printf("%i percent done\n", f / 10);
		}
	}


	int dictionarySize = 200;
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}


void CEssaiOpencvDlg::OnBnClickedButton7() //train
{
	string Nom[4] = { "accordion","airplanes","anchor","ant" };
	
	Mat dictionary;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);
	bowDE.setVocabulary(dictionary);
	char* filename = new char[100];
	char* imageTag = new char[10];
	string name;
	FileStorage fs1("descriptor.yml", FileStorage::WRITE);
	
	Mat trainData;
	Mat classData;
	CvSVM SVM;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 1.0; 
	params.term_crit = cvTermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);
	params.C = 0.1; 

	for (int i = 0; i < 4; i++) {
		for (int f = 1; f <= 10; f++)
		{
			sprintf(filename, "\\image_%04d.jpg", f);
			string nomf = ".\\" + Nom[i] + filename;
			
			Mat img = imread(nomf.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			vector<KeyPoint> keypoints;
			detector->detect(img, keypoints);
			Mat bowDescriptor;
			bowDE.compute(img, keypoints, bowDescriptor);
			trainData.push_back(bowDescriptor);
			classData.push_back(i);
			sprintf(imageTag, "img_%04d");
			name = Nom[i];

		}
	}
	
	SVM.train_auto(trainData, classData, Mat(), Mat(), params);
	SVM.save("model.xml");
	
	fs1.release();
}


void CEssaiOpencvDlg::OnBnClickedButton8() // test
{
	string Nom[4] = { "accordion","airplanes","anchor","ant" };

	Mat dictionary;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();

	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	bowDE.setVocabulary(dictionary);
	char* filename = new char[100];
	char* imageTag = new char[10];
	string name;
	FileStorage fs1("descriptor.yml", FileStorage::WRITE);

	Mat testData;
	Mat classData;
	CvSVM SVM;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.gamma = 1.0; 
	params.term_crit = cvTermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);
	params.C = 0.1; 
	string nomf = "ant\\image_0025.jpg";


	Mat img = imread(nomf.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);
	Mat bowDescriptor;
	bowDE.compute(img, keypoints, bowDescriptor);
	testData.push_back(bowDescriptor);
	int res;
	SVM.load("model.xml");
	res = SVM.predict(testData, false);


	char tab[100];
	string var = Nom[res];
	strcpy(tab, var.c_str());
	OutputDebugString(tab);
}


void CEssaiOpencvDlg::OnEnChangeEdit1()
{

}


void CEssaiOpencvDlg::OnBnClickedButton2()
{
	char tab[100];
	string var = "Desired_Text_String";
	strcpy(tab, var.c_str());
	OutputDebugString(tab);
}
