
function [corners,chessboards]  = demo(fname)

I = imread(fname);
corners = findCorners(I,0.01,1);
chessboards = chessboardsFromCorners(corners);

end
