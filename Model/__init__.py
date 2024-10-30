
from .MLP import *

model_dict = {'MLP':MLPM}




params_dict = {'LSTM':{'pass':{"input_features":2048,"dhidden" : 2048},'resnet':{"input_features":2048,"dhidden" : 2048},'densenet':{"input_features":1024,"dhidden" : 1024},'resnext':{"input_features":2048,"dhidden" : 2048},'open_clip':{"input_features":768,"dhidden" : 768 },'googlenet':{"input_features":1024,"dhidden":1024},'dinovit':{"input_features":768,"dhidden" : 768},'vit':{"input_features":1000,"dhidden" : 1000}}  , 'Transformer':{'pass': {
            "input_dims": 2048,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False
        },'resnet':{"input_dims": 2048,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False},
            'resnext': {"input_dims": 2048,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False}, 'densenet':{"input_dims": 1024,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False}
            ,'open_clip': {"input_dims": 768,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False},
        'dinovit':{"input_dims": 768,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False},'googlenet':{"input_dims": 1024,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False},'vit':{"input_dims": 1000,
            "transformer_dims": 512,
            "feedforward_dims": 1000,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False},'googlenet':{"input_dims": 1024,
            "transformer_dims": 512,
            "feedforward_dims": 1024,
            "pos_enc": [
                "absolute"
            ],
            "dropout": 0.5,
            "enable_scale": False,
            "depth": 1,
            "heads": [
                8
            ],
            "mask_params": [
                [
                    
                ]
            ],
            "skip_linear": False,
            "skip_att": False}}, 'MLP':{'pass':{"input_dims":2048,"feedforward_dims":1024},'resnet':{"input_dims":2048,"feedforward_dims":1024},'densenet':{"input_dims":1024,"feedforward_dims":512},'resnext':{"input_dims":2048,"feedforward_dims":1024},'open_clip':{"input_dims":768,"feedforward_dims" : 768 },'googlenet':{"input_dims":1024,"feedforward_dims":512},'dinovit':{"input_dims":768,"feedforward_dims" : 768},'vit':{"input_dims":1000,"feedforward_dims" : 1000}},
            
            
            '1DCNN':{'vit':{"input_features":1000,"dhidden" : 1000},'dinovit':{"input_features":768,"dhidden" : 768},'googlenet':{"input_features":1024,"dhidden":1024},'resnet':{"input_features":2048,"dhidden" : 2048},'densenet':{"input_features":1024,"dhidden" : 1024},'open_clip':{"input_features":768,"dhidden" : 768},'resnext':{"input_features":2048,"dhidden" : 2048},'pass':{"input_features":2048,"dhidden" : 2048}}
            }