
import sys
import os
import glob
import json
import argparse
import cProfile
from utils.utils import calc_mean_score, calc_max_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator

def make_dir(p):
    if os.path.exists(p):  # 判断文件夹是否存在
        # shutil.rmtree(p)        # 删除文件夹
        pass
    else:
        os.mkdir(p)  # 创建文件夹

def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        # img_id = img_path
        samples.append({'image_id': img_id})

    return samples


def predict(model, prepreocess_f, image_source, img_format):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, prepreocess_f(),
                                       img_format=img_format, img_load_dims=[224, 224])

    print(data_generator)
    preds = model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)

    return preds, samples


def predict_init_nima(base_model_name, weights_file):
    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)
    return nima.nima_model, nima.preprocessing_function

def profile_test(model, prepreocess_f, image_source, img_format, times):
    for i in range(times):
        predict(model, prepreocess_f, image_source, img_format)


def pprint_score(json_obj, img_src_dir, img_dst_dir, img_format='jpg'):
    make_dir(img_dst_dir)

    ind_score = {}
    for item in json_obj:
        ind_score[item['image_id']] = item['mean_score_prediction']

    import operator
    sorted_x = sorted(ind_score.items(), key=operator.itemgetter(1), reverse=True)

    i = 0
    for name, s in sorted_x:
        i += 1
        # print("%2d.jpg , %.4f, %.4f" % (i, s[0], s[1]))
        print("%s,%.4f" % (name, s))
        img_src = os.path.join(img_src_dir, name + '.' + img_format)
        img_dst = os.path.join(img_dst_dir, '{:0>5d}_{:0>3d}'.format(i, int(s*1000)) + '_' + name + '.' + img_format)

        import shutil
        try:
            shutil.copyfile(img_src, img_dst)
        except:
            print('[ERROR] copy [%s] to [%s] error' % (img_src, img_dst))
            print("Unexpected error:", sys.exc_info()[1])
            continue
        

import timeit
def main(base_model_name, weights_file, image_source, image_dst, predictions_file, performance_test, img_format='jpg'):
    # init model
    model, prepreocess_f = predict_init_nima(base_model_name, weights_file)
    seup_str = '''
from __main__ import predict
    '''

    print('IMAGE src: %s' % image_source)
    if performance_test is not None:
        tres = timeit.timeit('predict(model=model, prepreocess_f=prepreocess_f, image_source=image_source, img_format=img_format)', 
                       setup=seup_str, number=21, globals=locals())
        print(tres)
        # cProfile.runctx('profile_test(model, prepreocess_f, image_source, img_format, times)', {'profile_test':profile_test},
        #        {'model':model, 'prepreocess_f':prepreocess_f, 'image_source':image_source, 'img_format':img_format, 'times':3})
        return

    # get predictions, 两个返回值按照索引一一对应，和文件名无关
    predictions, samples = predict(model, prepreocess_f, image_source, img_format)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        # print("ind: %d, image_id: %s" % (i, sample['image_id']))
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])
        sample['max_score_prediction'] = int(calc_max_score(predictions[i]))

    pprint_score(samples, image_source, image_dst)
    # print(json.dumps(samples, indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-id', '--image-dst', help='image directory to store oredered image', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    parser.add_argument('-perf', '--performance-test', help='performance test', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)

