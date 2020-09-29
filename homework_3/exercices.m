clear all
close all

# T_1
img = imread('Figuras2/mammogram.bmp');
imfinfo Figuras2/mammogram.bmp;
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
