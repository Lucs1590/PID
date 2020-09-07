
clear all
close(gcf)

prompt = {'Enter the image name'};
dlgtitle = 'File Name';
definput = {'cameraman.tif'};
%opts.Interpreter = 'tex';
image_name = inputdlg(prompt,dlgtitle,[1 50],definput);
I = imread(image_name{1}) ;


BW1 = edge(I,'prewitt');
BW2 = edge(I,'sobel');
BW3 = edge(I,'roberts');
BW4 = edge(I,'canny');
BW5 = edge(I,'log');
subplot(2,3,1) ; imshow(I) ; title('Imagem Original')
subplot(2,3,2) ; imshow(BW1) ; title('Prewitt')
subplot(2,3,3) ;  imshow(BW2)  ; title('Sobel')
subplot(2,3,4) ; imshow(BW3) ; title('Roberts')
subplot(2,3,5) ;  imshow(BW4) ;  title('Canny')
subplot(2,3,6) ;  imshow(BW5) ;  title('Laplacian-Of-Gaussian')
pause

close(gcf)

I_mean = imboxfilt(I,5);

BW1 = edge(I_mean,'prewitt');
BW2 = edge(I_mean,'sobel');
BW3 = edge(I_mean,'roberts');
BW4 = edge(I_mean,'canny');
BW5 = edge(I_mean,'log');
subplot(2,3,1) ; imshow(I_mean) ; title('Imagem Original')
subplot(2,3,2) ; imshow(BW1) ; title('Prewitt')
subplot(2,3,3) ;  imshow(BW2)  ; title('Sobel')
subplot(2,3,4) ; imshow(BW3) ; title('Roberts')
subplot(2,3,5) ;  imshow(BW4) ;  title('Canny')
subplot(2,3,6) ;  imshow(BW5) ;  title('Laplacian-Of-Gaussian')
