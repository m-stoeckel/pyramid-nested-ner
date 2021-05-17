import json

from pyramid_nested_ner.data import DataPoint


class SentenceWindowDataPoint(DataPoint):

    def __init__(self, text, entities=None, pre_data_points=None, post_data_points=None):
        super(SentenceWindowDataPoint, self).__init__(text, entities)
        self.pre_data_points = pre_data_points
        self.post_data_points = post_data_points

    def _serialize(self):
        return {
            'text': self.text,
            'entities': [
                {
                    'name': entity.name,
                    'value': entity.value,
                    'start': entity.start,
                    'stop': entity.stop
                }
                for entity in self.entities
            ],
            'pre_data_points': [dp.serialize() for dp in self.pre_data_points],
            'post_data_points': [dp.serialize() for dp in self.post_data_points]
        }

    def __str__(self):
        return json.dumps(self._serialize(), indent=2)

    def __repr__(self):
        return json.dumps(self._serialize(), indent=2)
