import json

from pyramid_nested_ner.data import DataPoint


class ContextWindowDataPoint(DataPoint):

    def __init__(self, text, entities=None, pre_data_points=None, post_data_points=None):
        super(ContextWindowDataPoint, self).__init__(text, entities)
        self.preceding_data = pre_data_points
        self.subsequent_data = post_data_points

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
            'preceding_data': [dp.serialize() for dp in self.preceding_data],
            'subsequent_data': [dp.serialize() for dp in self.subsequent_data]
        }

    def __str__(self):
        return json.dumps(self._serialize(), indent=2)

    def __repr__(self):
        return json.dumps(self._serialize(), indent=2)
