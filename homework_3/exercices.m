clear all
close all

# T_1
img = imread('images/mammogram.bmp');
imfinfo images/mammogram.bmp;
figure(1)
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
imhist(img);

# E_1 e E_2
function my_histogram = MyHisto(img)
  [h w] = size(img);
  flat_image = img(:);
  aux = zeros(256,1);
  
  for i = 1:256
    aux(i,1) = sum(flat_image == i) / size(flat_image)(1);
  endfor;
  my_histogram = aux;
endfunction;
# hist = MyHisto(img);
hist = imhist(img, 256);
figure(2)
# subplot(1,3,1);

bar(hist,0.1);
title("Bar histogram");
xlabel("Px. value");
ylabel("Count");
text(32,8800,"Highest value")
axis([0 500 0 10000])


figure(3)
# subplot(1,3,2);
stem(hist, "linewidth", 0.1);
title("Stem histogram");
xlabel("Px. value");
ylabel("Count");
text(32,8800,"Highest value")
axis([0 350 0 10500])

figure(4)
# subplot(1,3,3);
plot(hist)
title("Curve continues histogram");
xlabel("Px. value");
ylabel("Count");
text(32,8800,"Highest value")
axis([0 300 0 10001])

# E_3
figure(5)
g1 = imadjust(img, [0 1], [1 0]); # inversa
imshow(g1)

figure(6)
g2 = imadjust(img, [0.5 0.75], [0 1]); # do 0 ao 0.5, recebe zero, do 0.5 ao 0.75 está entre 0 e 1, e acima de 0.75 recebe 1
imshow(g2)

figure(7)
g3 = imadjust(img, [ ], [ ], 2); # If gamma is greater than 1, then imadjust weights the mapping toward lower (darker) output values.
imshow(g3)

# T_2 e E_4
img2 = imread('images/polem.bmp');
figure(8)
imshow(img2)

figure(9)
imhist(img2)
ylim('auto')

img2_eq = histeq(img2,256);
figure(10)
imshow(img2_eq)

figure(11)
imhist(img2_eq)
ylim('auto')

# E_5
figure(12)
imshow(img2)

figure(13)
imhist(img2)
ylim('auto')

hnorm = imhist(img2)./numel(img2);
cdf = cumsum(hnorm);
x = linspace(0, 1, 256);
figure(14)
plot(x,cdf)
axis([0 1 0 1])

set(gca, 'xtick', 0:.2:1)
set(gca, 'ytick', 0:.2:1)
xlabel('Valores de Intensidade de Entrada', 'fontsize', 9)
ylabel('Valores de Intensidade de Saída', 'fontsize', 9)
text(0.18, 0.5, 'Função de Transfomação', 'fontsize', 9)
