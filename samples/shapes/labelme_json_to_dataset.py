import argparse
import json
import os
import os.path as osp
import warnings
import os.path
import subprocess
import numpy as np
import PIL.Image
import yaml
import cv2
import yaml
from labelme import utils


def main():

    json_file = 'C:/Users/QJ/Desktop/hh/total'
    list = os.listdir(json_file)
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.img_b64_to_array(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            out_dir=json_file+"/"+out_dir
            if not osp.exists(out_dir):
                os.mkdir(out_dir)

            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            # PIL.Image.fromarray(lbl).save()
            labelpath = osp.join(out_dir, 'label.png')
            # PIL.Image.fromarray(lbl).save(labelpath)
            # opencvimg16 = cv2.imread(labelpath)
            # opencvimg.convertTo(opencvimg6,)
            lbl8u=np.zeros((lbl.shape[0],lbl.shape[1]),dtype=np.uint8)
            for i in range(lbl.shape[0]):
                for j in range(lbl.shape[1]):
                    lbl8u[i,j]=lbl[i,j]
            PIL.Image.fromarray(lbl8u).save(labelpath);
            # Alllabelpath="%s"

        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in lbl_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=lbl_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.dump(info, f, default_flow_style=False)

        fov=open(osp.join(out_dir,'info.yaml'),'w')
        for key in info:
            fov.writelines(key)
            fov.write(':\n')
        for k,v in lbl_names.items():
            fov.write('  ')
            fov.write(k)
            fov.write(':\n')

        fov.close()
        print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
