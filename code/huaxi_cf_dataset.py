from __future__ import print_function
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np
import xlrd
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def normalize(slice):
    '''
        input: unnormalized slice
        OUTPUT: normalized clipped slice
    '''
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        return tmp


def fliplr(slice):
    if np.random.randint(0, 3) < 1:
        np.fliplr(slice)
    return slice


def move(slice):
    if np.random.randint(0, 3) < 1:
        x = np.random.randint(-9, 10)
        y = np.random.randint(-9, 10)
        z = np.random.randint(-2, 3)
        if x < 0 and y < 0 and z < 0:
            slice[:, z:-1, x:-1, y:-1] = 0
        if x >= 0 and y < 0 and z < 0:
            slice[:, z:-1, 0:x, y:-1] = 0
        if x < 0 and y >= 0 and z < 0:
            slice[:, z:-1, x:-1, 0:y] = 0
        if x < 0 and y < 0 and z >= 0:
            slice[:, 0:z, x:-1, y:-1] = 0
        if x >= 0 and y >= 0 and z < 0:
            slice[:, z:-1, 0:x, 0:y] = 0
        if x < 0 and y >= 0 and z >= 0:
            slice[:, 0:z, x:-1, 0:y] = 0
        if x >= 0 and y < 0 and z >= 0:
            slice[:, 0:z, 0:x, y:-1] = 0
        if x >= 0 and y >= 0 and z >= 0:
            slice[:, 0:z, 0:x, 0:y] = 0
    return slice


def cilp(slice):
    if np.random.randint(0, 3) < 1:
        fl = np.random.randint(0, 3)
        slice[0] = np.flip(slice[0], fl)
    return slice


def load_3dnpy(x_path):
    data, label = np.load(x_path, allow_pickle=True)
    # y = [1.]
    # if label == [1, 0]:
    #     y = [[1., 0.], [0., 1.]]
    cf = load_cf_form_xl(x_path)
    if '1958156' in x_path or '20291604' in x_path or '17428613' in x_path or '21052360' in x_path:
        print(x_path, cf)
    #data = normalize(data)
    return data, cf, label


class MyDataset(Dataset):
    def __init__(self, root, transform=True):
        self.image_files = np.array(root)
        self.transform = transform

    def __getitem__(self, index):  # 返回的是tensor

        x, cf, y = load_3dnpy(self.image_files[index])
        #print(y)
        # x = normalize(x, self.transform)
        x = x.transpose(2, 0, 1)
        x = x[np.newaxis, :, :, :]
        img = normalize(x)
        #print(img)
        if self.transform:
            img = fliplr(img)
            img = move(img)
            img = cilp(img)
        return torch.FloatTensor(img), torch.FloatTensor(cf), torch.FloatTensor(y)

    def __len__(self):
        return len(self.image_files)


def parse_xlsx(path, sheet, column_names):
    data = pd.ExcelFile(path)
    result = {}
    final_df = []
    for s in sheet:
        sheet_df = data.parse(s)
        final_df.append(sheet_df)
        # targ_data = sheet_df[column_names[1:]]
        # id_rows = sheet_df[column_names[0]]
        # targ_data = preprocessing(targ_data)
        # id_rows = id_rows.tolist()
        # for idx, row in targ_data.iterrows():
        #     result[id_rows[idx]] = row.values.tolist()
    final_df = pd.concat(final_df, ignore_index=True, sort=False)
    targ_data = final_df[column_names[1:]]
    targ_data = preprocessing(targ_data)
    id_rows = final_df[column_names[0]]
    for idx, row in targ_data.iterrows():
        result[id_rows[idx]] = row.values.tolist()
    return result


def preprocessing(df):
    """
    预处理，去除每一列的空值，并将非数值转化为数值型数据，分两步
    1. 如果本列含有null。
        - 如果是number类型
            如果全为空，则均置零；
            否则，空值的地方取全列平均值。
        - 如果不是number类型
            将空值置NA
    2. 如果本列不是数值型数据，则用label encoder转化为数值型
    :param df: dataframe
    :return: 处理后的dataframe
    """

    def process(c):
        if c.isnull().any().any():
            if np.issubdtype(c.dtype, np.number):
                new_c = c.fillna(c.mean())
                if new_c.isnull().any().any():
                    return pd.Series(np.zeros(c.size))
                return new_c
            else:
                return pd.Series(LabelEncoder().fit_transform(c.fillna("NA").values))
        else:
            if not np.issubdtype(c.dtype, np.number):
                return pd.Series(LabelEncoder().fit_transform(c.values))
        return c

    pre_df = df.copy()
    return pre_df.apply(lambda col: process(col))





def load_cf_form_xl(file):
    pid = file.split('/')[-1][:-4]
    xlsx_path = '/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/huaxi/受试者病理及检验数据.xlsx'

    class_names = ['MSS', 'MSI-H']
    # column_names = ['ID号', '性别', '年龄', '分化程度', 'T分期', 'ki-67(%)', 'CEA', 'CA19-9']
    column_names = ['ID号', '分化程度', 'T分期', 'ki-67(%)', 'CEA', 'CA19-9']
    ext_vector = parse_xlsx(xlsx_path, class_names, column_names)
    # print(ext_vector)
    # print(pid)
    cf = np.array(ext_vector[int(pid)])
    # 数据归一化
    max = np.array([1.0, 2.0, 10.0, 4.94, 1.18])
    min = np.array([1.0, 2.0, 10.0, 4.94, 1.18])
    for key in ext_vector.keys():
        # print(ext_vector[key)
        max = np.where(max > np.array(ext_vector[key]), max, np.array(ext_vector[key]))
        min = np.where(min > np.array(ext_vector[key]), np.array(ext_vector[key]), min)

    cf = ((cf-min)/(max - min) + 1)
    # print(cf)
    # exit()
    return np.log(cf)

# xlsx_path = '/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/huaxi/受试者病理及检验数据.xlsx'
#
# class_names = ['MSS', 'MSI-H']
#     # column_names = ['ID号', '性别', '年龄', '分化程度', 'T分期', 'ki-67(%)', 'CEA', 'CA19-9']
# column_names = ['ID号', '分化程度', 'T分期', 'ki-67(%)', 'CEA', 'CA19-9']
# ext_vector = parse_xlsx(xlsx_path, class_names, column_names)






def load_label_form_xl(xl_path):
    reads = xlrd.open_workbook(xl_path)
    label_2018 = []
    label_2019 = []
    for row in range(reads.sheet_by_index(0).nrows):
        if reads.sheet_by_index(0).cell(row, 4).value == 'CT':
            label_2018.append(int(reads.sheet_by_index(0).cell(row, 0).value))
            label_2018.append(reads.sheet_by_index(0).cell(row, 8).value)
            label_2018.append(reads.sheet_by_index(0).cell(row, 9).value)
    for row in range(reads.sheet_by_index(1).nrows):
        if reads.sheet_by_index(1).cell(row, 4).value == 'CT':
            label_2019.append(int(reads.sheet_by_index(1).cell(row, 0).value))
            label_2019.append(reads.sheet_by_index(1).cell(row, 8).value)
            label_2019.append(reads.sheet_by_index(1).cell(row, 9).value)

    return np.array(label_2018), np.array(label_2019)


def load_data(file, xl_path, file_list):
    label_2018, label_2019 = load_label_form_xl(xl_path)
    item_2018 = label_2018[0::3]
    item_2019 = label_2019[0::3]
    label_2018_s = label_2018[1::3]
    label_2019_s = label_2019[1::3]
    label_2018_m = label_2018[2::3]
    label_2019_m = label_2019[2::3]
    label = [1, 0, 0]
    year = file.split('/')[-4][:4]
    item = file.split('/')[-3]

    if year == '2018':
        item_index = list(item_2018).index(item)
        if label_2018_s[item_index] == '＋':
            label[0] = 0
            label[1] = 1
        if label_2018_m[item_index] == '＋':
            label[0] = 0
            label[2] = 1
    if year == '2019':
        item_index = list(item_2019).index(item)
        if label_2019_s[item_index] == '＋':
            label[0] = 0
            label[1] = 1
        if label_2019_m[item_index] == '＋':
            label[0] = 0
            label[2] = 1
    slices = load_dcm(file_list)
    print(slices.shape)
    slices = np.transpose(slices, (1, 2, 0))
    print(slices.shape)
    # slices = slices.transpose()
    max_x, min_x, max_y, min_y, max_z, min_z = load_mask(file)
    # print(max_x-min_x, max_y-min_y, max_z-min_z)
    x, y, z = 0, 0, 0
    if max_x - min_x < 96:
        x = int((96 - (max_x - min_x)) / 2 + 0.5)
        if max_x - min_x == 0:
            x = 1
    if max_y - min_y < 96:
        y = int((96 - (max_y - min_y)) / 2 + 0.5)
        if max_y - min_y == 0:
            y = 1
    if max_z - min_z < 16:
        z = int((16 - (max_z - min_z)) / 2 + 0.5)
        if max_z - min_z == 0:
            # print(max_z, min_z)
            z = -16

    if x == 0 and y == 0 and z == 0:
        data = slices[min_x:max_x, min_y:max_y, min_z:max_z]
    if x != 0 and y == 0 and z == 0:
        data = slices[min_x - x: 96 + min_x - x, min_y:max_y, min_z:max_z]
    if x == 0 and y != 0 and z == 0:
        data = slices[min_x:max_x, min_y - y: 96 + min_y - y, min_z:max_z]
    if x == 0 and y == 0 and z != 0:
        z_l, z_r = -1, -1
        if min_z - z < 0:
            z_l = 0
        if 16 + min_z - z > slices.shape[-1]:
            z_r = slices.shape[-1]

        if z_l == -1 and z_r > -1:
            data = slices[min_x:max_x, min_y:max_y, z_r - 16: z_r]
        elif z_l > -1 and z_r > -1:
            data = slices[min_x:max_x, min_y:max_y, z_l: z_r]
        elif z_l > -1 and z_r == -1:
            data = slices[min_x:max_x, min_y:max_y, z_l: 16 + z_l]
        else:
            data = slices[min_x:max_x, min_y:max_y, min_z - z: 16 + min_z - z]
        # data = slices[min_x:max_x, min_y:max_y, min_z - z: 16 + min_z - z]
    if x != 0 and y != 0 and z == 0:
        data = slices[min_x - x: 96 + min_x - x, min_y - y: 96 + min_y - y, min_z:max_z]
    if x != 0 and y == 0 and z != 0:
        z_l, z_r = -1, -1
        if min_z - z < 0:
            z_l = 0
        if 16 + min_z - z > slices.shape[-1]:
            z_r = slices.shape[-1]
        if z_l > -1 and z_r > -1:
            data = slices[min_x - x: 96 + min_x - x, min_y:max_y, z_l: z_r]
        elif z_l > -1 and z_r == -1:
            data = slices[min_x - x: 96 + min_x - x, min_y:max_y, z_l: 16 + z_l]
        elif z_l == -1 and z_r > -1:
            data = slices[min_x - x: 96 + min_x - x, min_y:max_y, z_r - 16: z_r]
        else:
            data = slices[min_x - x: 96 + min_x - x, min_y:max_y, min_z - z: 16 + min_z - z]

    if x == 0 and y != 0 and z != 0:
        # data = slices[min_x:max_x, min_y - y: 96 + min_y - y, min_z - z: 16 + min_z - z]
        z_l, z_r = -1, -1
        if min_z - z < 0:
            z_l = 0
        if 16 + min_z - z > slices.shape[-1]:
            z_r = slices.shape[-1]
        if z_l > -1 and z_r > -1:
            data = slices[min_x:max_x, min_y - y: 96 + min_y - y, z_l: z_r]
        elif z_l > -1 and z_r == -1:
            data = slices[min_x:max_x, min_y - y: 96 + min_y - y, z_l: 16 + z_l]
        elif z_l == -1 and z_r > -1:
            data = slices[min_x:max_x, min_y - y: 96 + min_y - y, z_r - 16: z_r]
        else:
            data = slices[min_x:max_x, min_y - y: 96 + min_y - y, min_z - z: 16 + min_z - z]
    if x != 0 and y != 0 and z != 0:
        z_l, z_r = -1, -1
        if min_z - z < 0:
            z_l = 0
        if 16 + min_z - z > slices.shape[-1]:
            z_r = slices.shape[-1]
        if z_l > -1 and z_r > -1:
            data = slices[min_x - x: 96 + min_x - x, min_y - y: 96 + min_y - y, z_l: z_r]
        elif z_l > -1 and z_r == -1:
            data = slices[min_x - x: 96 + min_x - x, min_y - y: 96 + min_y - y, z_l: 16 + z_l]
        elif z_l == -1 and z_r > -1:
            data = slices[min_x - x: 96 + min_x - x, min_y - y: 96 + min_y - y, z_r - 16: z_r]
        else:
            data = slices[min_x - x: 96 + min_x - x, min_y - y: 96 + min_y - y, min_z - z: 16 + min_z - z]
        # data = slices[min_x - x: 96 + min_x - x, min_y - y: 96 + min_y - y, min_z - z: 16 + min_z - z]
    # print('data',data.shape)
    # if data.shape[-1]<16:
    #     print("file name is:",max_z, min_z, file,data.shape)
    data = np.resize(data, (96, 96, 16))
    data.resize(96, 96, 16)
    data_label = []
    data_label.append(data)
    data_label.append(label)
    data_label = np.array(data_label)
    fla = np.random.randint(5)
    if fla > 0:
        f = '/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/gastric_cancer/train/' + file.split('/')[
            -4] + '_' + file.split('/')[-3] + '.npy'
    else:
        f = '/media/stianci/360b04be-f526-4e91-aad3-85535b17af3e/3d_data/gastric_cancer/test/' + file.split('/')[
            -4] + '_' + file.split('/')[-3] + '.npy'

    np.save(f, data_label)
    # print(data_label[0].shape, data_label[1], file)
    # print(data.shape)

    # return np.array(label).astype(np.uint8)


def load_mask(file):
    mask = nib.load(file).get_data()
    # print(mask.shape)
    bund = np.argwhere(mask == 1)
    max_x, min_x, max_y, min_y, max_z, min_z = bund[:, 0].max(), bund[:, 0].min(), bund[:, 1].max(), bund[:,
                                                                                                     1].min(), bund[:,
                                                                                                               2].max(), bund[
                                                                                                                         :,
                                                                                                                         2].min()
    return max_x, min_x, max_y, min_y, max_z, min_z


def load_dcm(file_list):
    slices = []
    for file in file_list:
        if 'nii' not in file:
            # print(dicom.read_file(file).pixel_array)
            slices.append(sitk.GetArrayFromImage(sitk.ReadImage(file))[0])
    #    slices.sort(key=lambda x: int(x.Im))
    return np.array(slices)



