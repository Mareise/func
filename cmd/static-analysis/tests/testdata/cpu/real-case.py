# Function
import torch
import torchvision.models as models
import json
import time

# Check device: use GPU if available, fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
model.to(device)
model.eval()

def new():
    """ New is the only method that must be implemented by a Function.
    The instance returned can be of any name.
    """
    return Function()


class Function:
    async def handle(self, scope, receive, send):
        try:
            # Create a big batch of random input
            batch_size = 128   # Big batch → makes inference more costly
            input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

            # Run a single inference and time it
            start_time = time.time()

            with torch.no_grad():
                output = model(input_tensor)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Results
            print(f"Device: {device}")
            print(f"Batch size: {batch_size}")
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            print(f"Output shape: {output.shape}")

            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [
                    [b'content-type', b'text/plain'],
                ],
            })
            await send({
                'type': 'http.response.body',
                'body': json.dumps({
                    "device": str(device),
                    "Elapsed time": elapsed_time,
                    "Output shape": list(output.shape)
                }).encode('utf-8')
            })

        except Exception as e:
            await send({
                'type': 'http.response.start',
                'status': 500,
                'headers': [
                    [b'content-type', b'text/plain'],
                ],
            })
            await send({
                'type': 'http.response.body',
                'body': f"Error processing request: {str(e)}".encode('utf-8')
            })
