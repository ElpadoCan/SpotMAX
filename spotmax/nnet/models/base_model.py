class BaseModel(object):

    def predict(self, **kwargs):
        """
        to overwrite
        """
        raise Exception('Method not already overwritten')

    def print_model(self, **kwargs):
        """
        to overwrite
        """
        raise Exception('Method not already overwritten')

    def train(self, **kwargs):
        """
        to overwrite
        """
        raise Exception('Method not already overwritten')

    def load(self, **kwargs):
        """
        to overwrite
        """
        raise Exception('Method not already overwritten')


