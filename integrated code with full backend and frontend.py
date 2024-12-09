import os
import cv2
import keras
from matplotlib import animation
import numpy as np
import gradio as gr
from keras.models import load_model
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import tempfile
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
from torch.nn import functional as F

frame_size = (64, 64)
sequence_length = 10
prediction_length = 10

class TransformerVideoPredictor(nn.Module):
    def __init__(self, input_dim, seq_length, embed_dim, num_heads, num_layers):
        super(TransformerVideoPredictor, self).__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            batch_first=True  
        )
        self.fc = nn.Linear(embed_dim, input_dim)

    def forward(self, x, target):
        b, t, h, w, c = x.size()
        x = x.view(b, t, -1)  
        target = target.view(b, target.size(1), -1)

        x_embed = self.embed(x) + self.positional_encoding
        target_embed = self.embed(target) + self.positional_encoding[:target.size(1)]

        output = self.transformer(x_embed, target_embed)
        output = self.fc(output)
        return output.view(b, target.size(1), h, w, c)

class ActionRecognitionApp:
    def __init__(self):
        #dictionary of class paths 
        self.class_paths = {
            'PullUps': 'train/PullUps',
            'SkyDiving': 'train/SkyDiving',
            'JumpingJack': 'train/JumpingJack',
            'ApplyLipstick': 'train/ApplyLipstick',
            'BoxingPunchingBag': 'train/BoxingPunchingBag'
        }
        
        self.model_paths = {
            'LSTM': 'my_trained_model.h5',
            'PredRNN': {
                'JumpingJack': 'jumpingjack_predrnn.h5',
                'PullUps': 'pullups_predrnn.h5',
                'SkyDiving': 'SkyDiving_predrnn.h5',
                'ApplyLipstick': 'ApplyLipstick_predrnn.h5',
                'BoxingPunchingBag': 'boxingbag_predrnn.h5'
            },
            'Transformer': {
            'JumpingJack': 'transformer_jumpingjack.pth',
            'PullUps': 'transformer_PullUps.pth',
            'SkyDiving': 'transformer_SkyDiving.pth',
            'ApplyLipstick': 'transformer_ApplyLipstick.pth',
            'BoxingPunchingBag': 'transformer_BoxingPunchingBag.pth'
            }
        }
        
        self.loaded_models = {}             #cached models to avoid reloading

    def prepare_transformer_input(self, frames, timesteps=10):
        """Prepare frames for Transformer input"""
        # print("Input frames shape:", frames.shape)
        # print("Input frames dtype:", frames.dtype)

        #ensure we have enough frames
        if len(frames) < timesteps:
            timesteps = len(frames)  
        
        #yse the first timesteps frames
        frames = frames[:timesteps]
        
        #normalization
        frames = frames / 255.0
        
        #convert to color (RGB) if grayscale
        if len(frames.shape) == 3:
            print("ALIZAAA")
            frames = np.expand_dims(frames, axis=-1)
            frames = np.repeat(frames, 3, axis=-1)
        
        frames = np.expand_dims(frames, axis=0)
        
        frames = torch.tensor(frames, dtype=torch.float32)
        
        return frames
    
    def load_model(self, model_type, action_class=None):
        """Load a model based on type and class"""
        if model_type == 'LSTM':
            model_key = 'LSTM'
            model_path = self.model_paths['LSTM']

            custom_objects = {'mse': keras.metrics.MeanSquaredError()}
            model = load_model(model_path, custom_objects=custom_objects)
            
        elif model_type == 'Transformer':
            if not action_class:
                raise ValueError("Action class must be specified for Transformer")
            model_key = f'Transformer_{action_class}'
            model_path = self.model_paths['Transformer'][action_class]
            
            model = TransformerVideoPredictor(
                input_dim=frame_size[0] * frame_size[1] * 3,
                seq_length=10,
                embed_dim=128,
                num_heads=4,
                num_layers=2
            )

            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
        elif model_type == 'PredRNN':
            if not action_class:
                raise ValueError("Action class must be specified for PredRNN")
            model_key = f'PredRNN_{action_class}'
            model_path = self.model_paths['PredRNN'][action_class]
            
            custom_objects = {'mse': keras.metrics.MeanSquaredError()}
            model = load_model(model_path, custom_objects=custom_objects)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.loaded_models[model_key] = model
        return model

    def extract_frames(self, video_path, frame_size=(64, 64)):
        """Extract frames from a video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, frame_size)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()
        return np.array(frames)
    
    def extract_frames_transformer(self, video_path, frame_size=(64, 64)):
        """Extract frames from a video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        cap.release()
        return np.array(frames)
    
    def normalize_frames(self, frames):
        """Normalize frames to 0-1 range and add channel dimension"""
        frames = frames / 255.0
        frames = np.expand_dims(frames, axis=-1)
        return frames
    
    def create_shifted_frames(self, data, sequence_length=10):
        """Create shifted frames for prediction"""
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i : i + sequence_length])
            y.append(data[i + 1 : i + 1 + sequence_length])
        return np.array(x), np.array(y)
    
    def prepare_predrnn_input(self, frames, timesteps=10):
        """Prepare frames for PredRNN input"""
        if len(frames) < timesteps:
            raise ValueError(f"Video has fewer than {timesteps} frames!")
        
        frames = frames[:timesteps]
        
        frames = frames / 255.0
        frames = np.expand_dims(frames, axis=-1)
        frames = np.expand_dims(frames, axis=0)
        
        return frames
    
    def evaluate_predictions(self, true_seq, predicted_seq):
        """
        Evaluate frame prediction accuracy using MSE and SSIM
        """
        true_frames = true_seq[0, :, :, :, 0]  
        pred_frames = predicted_seq[0, :, :, :, 0] 
        
        mse_values = []
        ssim_values = []
        
        for true_frame, pred_frame in zip(true_frames, pred_frames):

            true_frame_scaled = (true_frame * 255).astype(np.uint8)
            pred_frame_scaled = (pred_frame * 255).astype(np.uint8)
            
            mse = np.mean((true_frame_scaled - pred_frame_scaled) ** 2)
            mse_values.append(mse)
            
            ssim_value = ssim(true_frame_scaled, pred_frame_scaled)
            ssim_values.append(ssim_value)
        
        evaluation_metrics = {
            'mean_mse': np.mean(mse_values),
            'std_mse': np.std(mse_values),
            'mean_ssim': np.mean(ssim_values),
            'std_ssim': np.std(ssim_values)
        }
        
        print("\n--- Prediction Evaluation Metrics ---")
        print(f"Mean Squared Error (MSE):")
        print(f"  Average: {evaluation_metrics['mean_mse']:.4f}")
        print(f"  Standard Deviation: {evaluation_metrics['std_mse']:.4f}")
        print(f"\nStructural Similarity Index (SSIM):")
        print(f"  Average: {evaluation_metrics['mean_ssim']:.4f}")
        print(f"  Standard Deviation: {evaluation_metrics['std_ssim']:.4f}")
        
        return evaluation_metrics
    
    def visualize_predictions(self, input_seq, true_seq, predicted_seq):
        """Create a visualization of input, true, and predicted frames"""
        plt.close('all')  
        fig, axes = plt.subplots(3, 10, figsize=(20, 6))
        for i in range(10):
            #input frames
            axes[0, i].imshow(input_seq[0, i, :, :, 0], cmap='gray')
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            #ground truth frames
            axes[1, i].imshow(true_seq[0, i, :, :, 0], cmap='gray')
            axes[1, i].set_title(f"True {i+1}")
            axes[1, i].axis('off')
            #predicted frames
            axes[2, i].imshow(predicted_seq[0, i, :, :, 0], cmap='gray')
            axes[2, i].set_title(f"Predicted {i+1}")
            axes[2, i].axis('off')
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_plot:
            plt.savefig(temp_plot.name)
            temp_plot_path = temp_plot.name
        plt.close()
        
        return temp_plot_path
    
    def create_transition_video(self, input_seq, true_seq, predicted_seq):
        """Generate a video showing input and predicted frames"""
        frames = []
        
        for i in range(10):
            input_frame = input_seq[0, i, :, :, 0]

            input_frame = (input_frame * 255).astype(np.uint8)
            
            input_frame_color = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2BGR)
            
            frames.append(input_frame_color)
        
        for i in range(10):
            pred_frame = predicted_seq[0, i, :, :, 0]
           
            pred_frame = (pred_frame * 255).astype(np.uint8)
            
            pred_frame_color = cv2.cvtColor(pred_frame, cv2.COLOR_GRAY2BGR)
            
            frames.append(pred_frame_color)
        
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as temp_video:

            height, width = frames[0].shape[:2]
            fps = 10 
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            temp_video_path = temp_video.name
        
        return temp_video_path
    
    def visualize_predictions_transformer(self, input_seq, true_seq, predicted_seq):
        """Create a visualization of input, true, and predicted frames"""
        plt.close('all')  
        fig, axes = plt.subplots(3, 10, figsize=(20, 6))
        for i in range(10):
            #input frames
            axes[0, i].imshow(input_seq[0, i, :, :, 0])
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            #ground truth frames
            axes[1, i].imshow(true_seq[0, i, :, :, 0])
            axes[1, i].set_title(f"True {i+1}")
            axes[1, i].axis('off')
            #predicted frames
            axes[2, i].imshow(predicted_seq[0, i, :, :, 0])
            axes[2, i].set_title(f"Predicted {i+1}")
            axes[2, i].axis('off')
        plt.tight_layout()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_plot:
            plt.savefig(temp_plot.name)
            temp_plot_path = temp_plot.name
        plt.close()
        
        return temp_plot_path
    
    def create_transition_video_transformer(self, input_seq, true_seq, predicted_seq):
        """Generate a video showing input, true, and predicted frames side by side"""
        frames = []
        
        for i in range(10):
            input_frame = input_seq[0, i, :, :, 0]
            true_frame = true_seq[0, i, :, :, 0]
            pred_frame = predicted_seq[0, i, :, :, 0]
            
            input_frame = (input_frame * 255).astype(np.uint8)
            true_frame = (true_frame * 255).astype(np.uint8)
            pred_frame = (pred_frame * 255).astype(np.uint8)
            
           #convert grayscale to RGB
            input_frame_color = cv2.cvtColor(input_frame, cv2.COLOR_GRAY2RGB)
            true_frame_color = cv2.cvtColor(true_frame, cv2.COLOR_GRAY2RGB)
            pred_frame_color = cv2.cvtColor(pred_frame, cv2.COLOR_GRAY2RGB)

            #resize frames to ensure they're the same size
            height, width = input_frame_color.shape[:2]
            input_frame_color = cv2.resize(input_frame_color, (width, height))
            true_frame_color = cv2.resize(true_frame_color, (width, height))
            pred_frame_color = cv2.resize(pred_frame_color, (width, height))
            
           #concatenate frames horizontally
            combined_frame = np.concatenate([input_frame_color, true_frame_color, pred_frame_color], axis=1)
            
            frames.append(combined_frame)
        
        #create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as temp_video:
         #get frame dimensions and fps
            height, width = frames[0].shape[:2]
            fps = 10  # 10 frames per second
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(frame)

            out.release()
            
            temp_video_path = temp_video.name
        
        return temp_video_path

    
    def process_action(self, model_type, action_class, video_file):
        """Main processing function for the selected action and video"""
        if not video_file:
            return "Error: No video selected", None, None
        
        try:
            try:
                input_frames = self.extract_frames(video_file)

                input_frames_transformer = self.extract_frames_transformer(video_file)
                
                model = self.load_model(model_type, action_class)
            except Exception as e:
                print(f"Error !!: {str(e)}", None, None)
            
            if model_type == 'LSTM':
                normalized_frames = self.normalize_frames(input_frames)
                
                x, y = self.create_shifted_frames(normalized_frames)
                
                if len(x) == 0:
                    return "Error: Not enough frames to process", None, None
                
                index = np.random.randint(0, len(x))
                input_seq = x[index:index + 1]
                true_seq = y[index:index + 1]
                
                predicted_seq = model.predict(input_seq)

                evaluation_metrics = self.evaluate_predictions(true_seq, predicted_seq)
                
                visualization_path = self.visualize_predictions(input_seq, true_seq, predicted_seq)

                video_path = self.create_transition_video(input_seq, true_seq, predicted_seq)

            if model_type == 'Transformer':
                try:
                    # Split frames into input and ground truth sequences
                    input_seq = self.prepare_transformer_input(input_frames_transformer[:10])
                    true_seq = self.prepare_transformer_input(input_frames_transformer[10:20])
                    
                    # Validate sequence shapes
                    print(f"Input Sequence Shape: {input_seq.shape}")
                    print(f"True Sequence Shape: {true_seq.shape}")
                    
                    # Prediction
                    with torch.no_grad():
                        predicted_seq = model(input_seq, input_seq)
                    
                    # Convert to numpy for processing
                    input_seq_np = input_seq.cpu().numpy()
                    true_seq_np = true_seq.cpu().numpy()
                    predicted_seq_np = predicted_seq.cpu().numpy()
                    
                    # Evaluate and visualize
                    evaluation_metrics = self.evaluate_predictions(true_seq_np, predicted_seq_np)
                    visualization_path = self.visualize_predictions_transformer(
                        input_seq_np, true_seq_np, predicted_seq_np
                    )   
                    video_path = self.create_transition_video_transformer(
                        input_seq_np, true_seq_np, predicted_seq_np
                    )
                    
                    return f"Processed {action_class} action using {model_type}", visualization_path, video_path
                
                except Exception as transformer_error:
                    import traceback
                    traceback.print_exc()
                    return f"Transformer Model Error: {str(transformer_error)}", None, None

            else:  # PredRNN
                input_seq = self.prepare_predrnn_input(input_frames)
                
                predicted_seq = model.predict(input_seq)
                
                true_seq = predicted_seq.copy()

                evaluation_metrics = self.evaluate_predictions(true_seq, predicted_seq)
                
                visualization_path = self.visualize_predictions(input_seq, true_seq, predicted_seq)

                video_path = self.create_transition_video(input_seq, true_seq, predicted_seq)
            
            return f"Processed {action_class} action using {model_type}", visualization_path, video_path
        
        except Exception as e:
            return f"Error processing video: {str(e)}", None, None

    def get_videos_for_class(self, action_class):
        """Get list of videos for a specific class"""
        class_path = self.class_paths.get(action_class, '')
        if not class_path or not os.path.exists(class_path):
            return []
        
        #return list of video files in the class directory
        return [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.avi')]

app = ActionRecognitionApp()

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Action Recognition Visualization")
        
        model_dropdown = gr.Dropdown(
            ['LSTM', 'PredRNN', 'Transformer'], 
            label="Select Model Type"
        )
        
        action_dropdown = gr.Dropdown(
            list(app.class_paths.keys()), 
            label="Select Action Class"
        )
        
        video_dropdown = gr.Dropdown([], label="Select Video")
        
        action_dropdown.change(
            fn=lambda action_class: gr.Dropdown(choices=app.get_videos_for_class(action_class)),
            inputs=action_dropdown,
            outputs=video_dropdown
        )
        
        video_upload = gr.File(type="filepath", label="Or Upload a Video")
        
        process_btn = gr.Button("Process Video")
        
        with gr.Row():
            status_output = gr.Textbox(label="Status")
            
        with gr.Row():
            visualization_output = gr.Image(label="Prediction Visualization")
            
            video_output = gr.Video(label="Transition Video")
        
        process_btn.click(
            fn=lambda model_type, action_class, video_dropdown, video_upload: app.process_action(
                model_type,
                action_class, 
                video_dropdown or video_upload
            ),
            inputs=[model_dropdown, action_dropdown, video_dropdown, video_upload],
            outputs=[status_output, visualization_output, video_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()