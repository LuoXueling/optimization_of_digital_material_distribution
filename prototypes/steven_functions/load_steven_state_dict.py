import torch
from picnn_model_SIAM_with_x0_concat import PICNN_network


a = torch.load("./prototypes/steven_functions/model_saved_step7.pth", weights_only=True)

input_dim = 2
output_dim = 1
hidden_number_layers_details = [30, 30]
input_dim1 = 1
hidden_number_layers_details1 = [5, 5]

device = "cuda"

model = PICNN_network(
    input_dim,
    output_dim,
    hidden_number_layers_details,
    input_dim1,
    hidden_number_layers_details1,
).to(device)

model.load_state_dict(a)
nc_weights = model.get_weights_nc()
ncfc_weights = model.get_weights_ncfc()
fc_weights = model.get_weights_fc()

print(nc_weights)
