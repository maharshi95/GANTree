from exp_context import ExperimentContext

hyperparam_file = 'hyperparams/toy_gnode/bcgan_3d.py'
ExperimentContext.set_context(hyperparam_file, 'gantree')

Model = ExperimentContext.Model
H = ExperimentContext.Hyperparams

from dataloaders.factory import DataLoaderFactory
from _tf.gan_tree import gan_tree

dl = DataLoaderFactory.get_dataloader(H.dataloader)

x_train, x_test = dl.get_data()

tree = gan_tree.GanTree('gan-tree', Model, x_test)
