from transformers import pipeline
from cache_manager import CacheManager

class ModelManager:
    def __init__(self, use_cache=True, cache_dir="./model_cache"):
        # Initialize models
        self.asr_model = None
        self.translator = None
        self.tts_model = None
        
        # Initialize cache manager
        self.use_cache = use_cache
        self.cache_manager = CacheManager(cache_dir) if use_cache else None
        
        # Model IDs
        self.asr_model_id = "openai/whisper-small" # "facebook/wav2vec2-large-960h-lv60-self"
        self.translator_model_id = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko" # "Helsinki-NLP/opus-mt-tc-big-en-ko"
        self.tts_model_id = "" # "facebook/mms-tts-kor"
        
    def load_models(self):
        """Load all required models with caching support"""
        self._load_asr_model()
        self._load_translator_model()
        self._load_tts_model()
        print("All models loaded successfully!")
        
    def _load_asr_model(self):
        """Load ASR model with caching"""
        print("Loading ASR model...")
        
        if self.use_cache:
            # Check if model is cached
            cached_path = self.cache_manager.get_model_path(self.asr_model_id)
            
            if cached_path:
                print(f"Loading ASR model from cache: {cached_path}")
                self.asr_model = pipeline(
                    "automatic-speech-recognition",
                    model=cached_path,
                    chunk_length_s=10
                )
            else:
                # Download and cache the model
                cached_path = self.cache_manager.cache_model(
                    self.asr_model_id, 
                    "automatic-speech-recognition"
                )
                
                if cached_path:
                    self.asr_model = pipeline(
                        "automatic-speech-recognition",
                        model=cached_path,
                        chunk_length_s=10
                    )
                else:
                    # Fallback to direct loading
                    self.asr_model = pipeline(
                        "automatic-speech-recognition",
                        model=self.asr_model_id,
                        chunk_length_s=10
                    )
        else:
            # Direct loading without cache
            self.asr_model = pipeline(
                "automatic-speech-recognition",
                model=self.asr_model_id,
                chunk_length_s=10
            )
            
    def _load_translator_model(self):
        """Load translator model with caching"""
        print("Loading translation model...")
        
        if self.use_cache:
            # Check if model is cached
            cached_path = self.cache_manager.get_model_path(self.translator_model_id)
            
            if cached_path:
                print(f"Loading translator model from cache: {cached_path}")
                self.translator = pipeline(
                    "translation", 
                    model=cached_path
                )
            else:
                # Download and cache the model
                cached_path = self.cache_manager.cache_model(
                    self.translator_model_id, 
                    "translation"
                )
                
                if cached_path:
                    self.translator = pipeline(
                        "translation", 
                        model=cached_path
                    )
                else:
                    # Fallback to direct loading
                    self.translator = pipeline(
                        "translation", 
                        model=self.translator_model_id
                    )
        else:
            # Direct loading without cache
            self.translator = pipeline(
                "translation", 
                model=self.translator_model_id
            )
            
    def _load_tts_model(self):
        """Load TTS model with caching"""
        print("Loading TTS model...")
        
        if self.use_cache:
            # Check if model is cached
            cached_path = self.cache_manager.get_model_path(self.tts_model_id)
            
            if cached_path:
                print(f"Loading TTS model from cache: {cached_path}")
                self.tts_model = pipeline(
                    "text-to-speech",
                    model=cached_path
                )
            else:
                # Download and cache the model
                cached_path = self.cache_manager.cache_model(
                    self.tts_model_id,
                    "text-to-speech"
                )
                
                if cached_path:
                    self.tts_model = pipeline(
                        "text-to-speech",
                        model=cached_path
                    )
                else:
                    # Fallback to direct loading
                    self.tts_model = pipeline(
                        "text-to-speech",
                        model=self.tts_model_id
                    )
        else:
            # Direct loading without cache
            self.tts_model = pipeline(
                "text-to-speech",
                model=self.tts_model_id
            )
        
    def transcribe_audio(self, audio_data):
        """Convert speech to text using HuggingFace ASR pipeline"""
        result = self.asr_model(audio_data)
        return result["text"]
    
    def translate_text(self, text):
        """Translate English text to Korean"""
        result = self.translator(text, max_length=400)
        return result[0]['translation_text']
    
    def generate_speech(self, text):
        """Generate audio from Korean text"""
        return self.tts_model(text)["audio"]
        
    def get_cache_info(self):
        """Get information about the model cache"""
        if self.use_cache:
            return self.cache_manager.get_cache_info()
        return {"error": "Caching is disabled"}
