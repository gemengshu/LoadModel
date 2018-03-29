function SplitDataset(image_path, gd_path, save_path, img_suffix, gd_suffix)
% PARAMETERS
% image_path  -- full directory of the original dataset
% gd_path     -- full directory of the ground truth data
% save_path   --
% img_suffix  -- suffix and format of the original image for saving, like '.png'
% gd_suffix   -- suffix and format of the ground truth image for saving,
% like '_mask.png' or 'up.png'
%
% NOTICE: all the directory should end with a slash '\'
%
% edited by mengshu
% 3/29/2018

train_path = strcat(save_path, '\train\');
test_path = strcat(save_path, '\test\');

if exist(train_path, 'dir') == 0
    mkdir(train_path);
end
if exist(test_path, 'dir') == 0
    mkdir(test_path);
end

% folder name
gd_folder = gd_path;

noise_folder = image_path;

% put all the images together and split them randomly
per = 0.75; % percentage of training dataset
files = dir(strcat(noise_folder,'*', img_suffix));

total_num = size(files,1);
training_num = floor(total_num * per);

rand_ind = randperm(total_num);

count = 1;

for i = 1 : training_num

    ind = rand_ind(i);
    file_name = files(ind).name;

    gd = imread([gd_folder, file_name]);
    noise = imread([noise_folder, file_name]);

    split_name = strsplit(file_name, '.');
    name = split_name{1};

    new_gd_name = strcat(name, gd_suffix);

    imwrite(gd, [train_path, new_gd_name]);
    imwrite(noise, [train_path, file_name]);

    count = count + 1;

    if mod(count, 100) == 0
        disp([num2str(count), ' images have been processed.']);
    end

end

for i = training_num + 1 : total_num

    ind = rand_ind(i);
    file_name = files(ind).name;

    gd = imread([gd_folder, file_name]);
    noise = imread([noise_folder, file_name]);

    split_name = strsplit(file_name, '.');
    name = split_name{1};

    new_gd_name = strcat(name, gd_suffix);

    imwrite(gd, [test_path, new_gd_name]);
    imwrite(noise, [test_path, file_name]);

    count = count + 1;

    if mod(count, 100) == 0
        disp([num2str(count), ' images have been processed.']);
    end
end
