clear all
close(gcf)


I = imread('pirate.png');

J = histeq(I);

imshowpair(I,J,'montage')
axis off

pause

close(gcf)

K = histeq(J);

imshowpair(J,K,'montage')
axis off

pause

close(gcf)

% Display a histogram of the original image.
subplot(1,2,1)
imhist(I,64)

% Display a histogram of the processed image.

subplot(1,2,2)
imhist(J,64)

pause

close(gcf)

