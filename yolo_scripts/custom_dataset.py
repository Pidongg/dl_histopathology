from ultralytics.data.dataset import YOLODataset
import albumentations as A

class CLAHEDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clahe_transform = A.Compose([
            A.CLAHE(
                clip_limit=1.0,
                tile_grid_size=(8, 8),
                p=0.5
            )
        ])
    
    def get_image(self, index):
        img = super().get_image(index)
        if self.augment:
            img = self.clahe_transform(image=img)['image']
        return img