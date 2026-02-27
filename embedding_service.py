from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None
    _model = None
    _loading = False  # Add this flag
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    
    def get_model(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
        if EmbeddingService._model is None and not EmbeddingService._loading:
            EmbeddingService._loading = True
            logger.info(f"Loading embedding model: {model_name}")
            EmbeddingService._model = SentenceTransformer(model_name,local_files_only=True)
            EmbeddingService._loading = False
        return EmbeddingService._model

# # Global instance
embedding_service = EmbeddingService()
#####################################

# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F
# import logging

# logger = logging.getLogger(__name__)

# class EmbeddingService:
#     _instance = None
#     _model = None
#     _tokenizer = None
#     _loading = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def get_model(self, model_name: str = 'google/embeddinggemma-300m'):
#         if EmbeddingService._model is None and not EmbeddingService._loading:
#             EmbeddingService._loading = True
#             logger.info(f"Loading embedding model: {model_name}")
#             EmbeddingService._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#             EmbeddingService._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#             EmbeddingService._loading = False
#         return self
    
#     def encode(self, texts):
#         if isinstance(texts, str):
#             texts = [texts]
        
#         inputs = EmbeddingService._tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = EmbeddingService._model(**inputs)
#             embeddings = outputs.last_hidden_state.mean(dim=1)
#             embeddings = F.normalize(embeddings, p=2, dim=1)
        
#         return embeddings.cpu().numpy()

# embedding_service = EmbeddingService()

# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F
# import logging

# logger = logging.getLogger(__name__)

# class EmbeddingService:
#     _instance = None
#     _model = None
#     _tokenizer = None
#     _loading = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def get_model(self, model_name: str = 'google/embeddinggemma-300m'):
#         if EmbeddingService._model is None and not EmbeddingService._loading:
#             EmbeddingService._loading = True
#             logger.info(f"Loading embedding model: {model_name}")
#             EmbeddingService._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#             EmbeddingService._model = AutoModel.from_pretrained(
#                 model_name, 
#                 trust_remote_code=True,
#                 torch_dtype=torch.float32,
#                 device_map="cuda" if torch.cuda.is_available() else "cpu"
#             )
#             EmbeddingService._loading = False
#         return self
    
#     def encode(self, texts):
#         if isinstance(texts, str):
#             texts = [texts]
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         inputs = EmbeddingService._tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = EmbeddingService._model(**inputs)
#             embeddings = outputs.last_hidden_state.mean(dim=1)
#             embeddings = F.normalize(embeddings, p=2, dim=1)
        
#         return embeddings.cpu().numpy()

# embedding_service = EmbeddingService()

################### EMBEDDING GEMMA #####################
# from sentence_transformers import SentenceTransformer
# import logging

# logger = logging.getLogger(__name__)

# class EmbeddingService:
#     _instance = None
#     _model = None
#     _loading = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instancel
    
#     def get_model(self, model_name: str = 'google/embeddinggemma-300m'):
#         if EmbeddingService._model is None and not EmbeddingService._loading:
#             EmbeddingService._loading = True
#             logger.info(f"Loading embedding model: {model_name}")
#             EmbeddingService._model = SentenceTransformer(model_name, trust_remote_code=True)
#             EmbeddingService._loading = False
#         return EmbeddingService._model

# # Global instance
# embedding_service = EmbeddingService()
# from FlagEmbedding import BGEM3FlagModel
# import logging

# logger = logging.getLogger(__name__)

# class EmbeddingService:
#     _instance = None
#     _model = None
#     _loading = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def get_model(self, model_name: str = 'BAAI/bge-m3'):
#         if EmbeddingService._model is None and not EmbeddingService._loading:
#             EmbeddingService._loading = True
#             logger.info(f"Loading embedding model: {model_name}")
#             EmbeddingService._model = BGEM3FlagModel(model_name, use_fp16=True,device='cuda:0')
#             EmbeddingService._loading = False
#         return EmbeddingService._model

# embedding_service = EmbeddingService()