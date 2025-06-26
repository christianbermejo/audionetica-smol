import time
import numpy as np
import os
from models import ModelManager
from audio_processor import AudioProcessor
from utils import normalize_audio, create_temp_wav_file

def main():
    print("Initializing Speech-to-Speech Translation Tool")
    
    # Check for cache directory in environment or use default
    cache_dir = os.environ.get("MODEL_CACHE_DIR", "./model_cache")
    use_cache = os.environ.get("USE_MODEL_CACHE", "1") == "1"
    
    # Initialize components
    model_manager = ModelManager(use_cache=use_cache, cache_dir=cache_dir)
    audio_processor = AudioProcessor()
    
    try:
        # Load models
        model_manager.load_models()
        
        while True:
            print("1. Start translation")
            print("2. Show cache information")
            print("3. Clear cache")
            print("4. Exit")
            choice = input("Select an option: ")
            
            if choice == "1":
                # Record audio
                print("Press Enter to start recording (10 seconds)...")
                input()
                recording_thread = audio_processor.start_recording(duration=10)
                recording_thread.join()  # Wait for recording to complete
                
                # Process audio
                audio_data = audio_processor.get_audio_data()
                if audio_data is None:
                    print("No audio recorded!")
                    continue
                    
                # Normalize audio
                audio_data = normalize_audio(audio_data)
                
                # Create temporary WAV file for the ASR model
                temp_wav = create_temp_wav_file(audio_data)
                
                # Speech recognition
                print("Transcribing audio...")
                transcription = model_manager.transcribe_audio(temp_wav)
                print(f"Transcription (English): {transcription}")
                
                # Translation
                print("Translating to Korean...")
                translation = model_manager.translate_text(transcription)
                print(f"Translation (Korean): {translation}")
                
                # Text to speech
                print("Generating speech...")
                try:
                    speech_audio = model_manager.generate_speech(translation)
                    audio_processor.play_audio(speech_audio)
                except Exception as e:
                    print(f"Error with HuggingFace TTS: {e}")
                    print("Falling back to gTTS...")
                    audio_processor.play_audio_from_tts(translation)
                    
            elif choice == "2":
                # Show cache information
                if use_cache:
                    cache_info = model_manager.get_cache_info()
                    print("\\nCache Information:")
                    print(f"Total models cached: {cache_info['model_count']}")
                    print(f"Total cache size: {cache_info['total_size_mb']:.2f} MB")
                    print("\\nCached models:")
                    for model_id, size in cache_info.get('models', {}).items():
                        print(f"- {model_id}: {size / (1024 * 1024):.2f} MB")
                else:
                    print("Caching is disabled")
                    
            elif choice == "3":
                # Clear cache
                if use_cache:
                    clear_all = input("Clear all models? (y/n): ").lower() == 'y'
                    if clear_all:
                        model_manager.cache_manager.clear_cache()
                    else:
                        print("\\nAvailable models:")
                        for i, model_id in enumerate(model_manager.cache_manager.cache_index["models"].keys()):
                            print(f"{i+1}. {model_id}")
                        model_idx = input("Enter model number to clear (or 0 to cancel): ")
                        try:
                            model_idx = int(model_idx)
                            if model_idx > 0:
                                model_id = list(model_manager.cache_manager.cache_index["models"].keys())[model_idx-1]
                                model_manager.cache_manager.clear_cache(model_id)
                        except (ValueError, IndexError):
                            print("Invalid selection")
                else:
                    print("Caching is disabled")
                
            elif choice == "4":
                break
            else:
                print("Invalid option!")
                
    except KeyboardInterrupt:
        print("\\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        audio_processor.cleanup()
        
if __name__ == "__main__":
    main()
