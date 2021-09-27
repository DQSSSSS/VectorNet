import config
import os
from network.vector_net import VectorNet, VectorNetWithPredicting
from DataLoader import *

def train_model(vector_net, training_data, save_name, eval_data=None, epochs=25, learning_rate=0.001, decayed_factor=0.3):
    data = VectorNetData(training_data)
    train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
    vector_net = vector_net.to(config.device)

    optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = vector_net(inputs["item_num"].to(config.device), 
                inputs["target_id"].to(config.device), inputs["polyline_list"])
            loss = loss_func(outputs, labels["future"].to(config.device))
            loss.backward()
            optimizer.step()
            print(epoch, i, loss.item())

        if (epoch + 1) % 5 == 0:
            learning_rate *= decayed_factor
            optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)

            if eval_data != None:
                # TODO
                pass

    torch.save(vector_net, os.path.join(config.model_save_path, save_name + '.model'))


if __name__ == '__main__':
    print("now device:", config.device)
    v_len = 9
    vector_net = VectorNetWithPredicting(v_len=v_len, time_stamp_number=30)
    training_data = get_random_data(50, v_len)

    train_model(vector_net, training_data, "random_model")

