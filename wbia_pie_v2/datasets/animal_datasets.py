# -*- coding: utf-8 -*-
from .coco_dataset import COCODataset
import yaml
import os.path as osp
import os


# This dynamically creats a COCODataset object from looking at a config file
class ConfigDataset(COCODataset):

    def __init__(self, config_fpath, **kwargs):
        conf = _read_and_validate_dataset_config(config_fpath)
        # name is the config filename without extension or containing folders
        id_attr = ['name']
        if conf['data'].get('use_viewpoint', True):
            id_attr.append('viewpoint')
        mode = kwargs.get('mode', 'train')
        super(ConfigDataset, self).__init__(
            name=os.path.splitext(os.path.split(config_fpath)[1])[0],
            dataset_url=conf['data'].get('dataset_url', None),
            dataset_dir=conf['data']['coco_dir'],
            split=conf['data'].get('split', f'{mode}2021'),
            crop=conf['data'].get('crop', True),
            resize=conf['data'].get('resize', True),
            imsize=min(conf['data']['height'], conf['data']['width']),
            train_min_samples=conf['data'].get('train_min_samples', 3),
            test_min_samples=conf['data'].get('test_min_samples', 3),
            viewpoint_list=conf['data'].get('viewpoint_list', ['left', 'right']),
            debug=conf['data'].get('debug', False),
            **kwargs
        )

    def _get_coco_db(self, split):
        """ Get database from COCO anntations """
        ann_file = osp.join(
            self.dataset_dir_orig,
#            '{}.coco'.format(self.name),
            '{}.json'.format(split),
        )
        if not osp.exists(ann_file):
            # below path is used on automated datasets
            ann_file = osp.join(
                self.dataset_dir,
                '{}.json'.format(split),
            )

        dataset = json.load(open(ann_file, 'r'))

        # Get image metadata
        imgs = {}
        if 'images' in dataset:
            for img in dataset['images']:
                imgs[img['id']] = img

        # Get annots for images from annotations and parts
        imgToAnns = defaultdict(list)
        if 'annotations' in dataset:
            for ann in dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)

        if 'parts' in dataset:
            for ann in dataset['parts']:
                imgToAnns[ann['image_id']].append(ann)

        image_set_index = list(imgs.keys())
        print('=> Found {} images in {}'.format(len(image_set_index), ann_file))

        # Get viewpoint annotations from a separate csv
        if self.viewpoint_csv is not None:
            uuid2view = np.genfromtxt(
                self.viewpoint_csv, dtype=str, skip_header=1, delimiter=','
            )
            print(
                '=> Found {} view annotations in {}'.format(
                    len(uuid2view), self.viewpoint_csv
                )
            )
            uuid2view = {a[0]: a[1] for a in uuid2view}
        else:
            uuid2view = None

        # Collect ground truth annotations
        gt_db = []
        for index in image_set_index:
            img_anns = imgToAnns[index]
            image_path = self._get_image_path(imgs[index]['file_name'])
            gt_db.extend(self._load_image_annots(img_anns, image_path, uuid2view))
        return gt_db

def _read_and_validate_dataset_config(config_fpath):
    with open(config_fpath, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    assert 'data' in conf and 'coco_dir' in conf['data'], 'Error: ConfigDataset needs data.coco_dir to be defined in config file.'
    return conf


class WhaleShark(COCODataset):
    def __init__(self, **kwargs):
        super(WhaleShark, self).__init__(
            name='whaleshark',
            dataset_dir='whaleshark',
            dataset_url='https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz',
            split='train2020',
            crop=True,
            resize=True,
            imsize=256,
            train_min_samples=5,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            **kwargs
        )


class WhaleSharkCropped(COCODataset):
    def __init__(self, **kwargs):
        super(WhaleSharkCropped, self).__init__(
            name='whaleshark_cropped',
            dataset_dir='whaleshark_cropped',
            dataset_url='https://www.dropbox.com/s/4tky8z0g4ob6qfx/coco.whaleshark_cropped.tar.gz?dl=1',
            split='test2021',
            viewpoint_csv='reid-data/whaleshark_cropped/original/whaleshark_cropped.coco/annotations/annotViewpointMap.csv',
            crop=False,
            resize=True,
            imsize=256,
            train_min_samples=5,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            **kwargs
        )


class MantaRayCropped(COCODataset):
    def __init__(self, **kwargs):
        super(MantaRayCropped, self).__init__(
            name='mantaray_cropped',
            dataset_dir='mantaray_cropped.coco',
            dataset_url=None,
            split='train2021',
            crop=False,
            resize=False,
            imsize=300,
            train_min_samples=5,
            id_attr=['name'],
            debug=False,
            **kwargs
        )


class GrayWhale(COCODataset):
    def __init__(self, **kwargs):
        super(GrayWhale, self).__init__(
            name='graywhale',
            dataset_dir='graywhale',
            dataset_url='',
            split='train_right',
            split_test='test_left',
            crop=True,
            flip_test=True,
            resize=True,
            imsize=256,
            train_min_samples=3,
            test_min_samples=3,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            excluded_names='____',
            **kwargs
        )


class HyenaBothsides(COCODataset):
    def __init__(self, **kwargs):
        super(HyenaBothsides, self).__init__(
            name='hyena_bothsides',
            dataset_dir='/data/db/_ibsdb/_ibeis_cache/hyena_bothsides',
            dataset_url=None,
            split='train2021',
            split_test='test2021',
            crop=True,
            resize=True,
            imsize=256,
            train_min_samples=3,
            test_min_samples=3,
            id_attr=['name', 'viewpoint'],
            debug=False,
            excluded_names='____',
            **kwargs
        )


class WildHorseFace(COCODataset):
    def __init__(self, **kwargs):
        super(WildHorseFace, self).__init__(
            name='wildhorse_face',
            dataset_dir='wildhorses_combined',
            dataset_url='',
            split='train2021',
            crop=True,
            flip_test=False,
            resize=True,
            imsize=300,
            train_min_samples=3,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['front'],
            debug=False,
            excluded_names='____',
            **kwargs
        )

class BearFace(COCODataset):
    def __init__(self, **kwargs):
        super(BearFace, self).__init__(
            name='bear_face',
            dataset_dir='/data/output',
            dataset_url='',
            split='v1_train_85_coco_amended',
            split_test='v1_val_15_coco_amended',
            crop=False,
            flip_test=False,
            resize=False,
            imsize=150,
            train_min_samples=10,
            test_min_samples=10,
            id_attr=['name'],
            viewpoint_list=['front'],
            debug=False,
            excluded_names='____',
            **kwargs
        )
