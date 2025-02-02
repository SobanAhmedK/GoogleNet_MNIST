function test_network(net, image)
    I = imread(image);  
    R = imresize(I, [224, 224]);

   
    if size(R, 3) == 1
        R = cat(3, R, R, R); 
    end    

    [Label, Probability] = classify(net, R);

    figure;
    imshow(I); 
    title(['Predicted: ', char(Label), ' - Probability: ', num2str(max(Probability), '%.6f')]);
end
