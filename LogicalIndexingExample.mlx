img = imread("file01.jpg");
imshow(img)
net = googlenet;
categorynames = net.Layers(end).ClassNames;

[pred,scores] = classify(net,img)

bar(scores)

highscores = scores > 0.01

bar(scores(highscores))
