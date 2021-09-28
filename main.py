import config
import os
from network.vector_net import VectorNet, VectorNetWithPredicting
from dataloader.random_dataloader import *
from loss_and_eval.evaluation import *
from loss_and_eval.loss import *

def train_model(vector_net, dataloader, loss_func, save_name, is_print_eval=True, is_print_test=False, epochs=25, learning_rate=0.001, decayed_factor=0.3):
    train_loader = dataloader.training_dataloader
    eval_loader = dataloader.eval_dataloader

    vector_net = vector_net.to(config.device)
    optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = vector_net(inputs["item_num"].to(config.device), 
                inputs["target_id"].to(config.device), inputs["polyline_list"])
            loss = loss_func(outputs, labels["future"].to(config.device))
            loss.backward()
            optimizer.step()
            if is_print_test:
                print("epoch:", epoch, "iteration:", i, "loss function:", loss)

        if (epoch + 1) % 5 == 0:
            learning_rate *= decayed_factor
            optimizer = torch.optim.Adam(vector_net.parameters(), lr=learning_rate)

            if is_print_eval:
                loss, ade, t = 0, 0, 0
                for i, data in enumerate(eval_loader):
                    inputs, labels = data
                    outputs = vector_net(inputs["item_num"].to(config.device), 
                        inputs["target_id"].to(config.device), inputs["polyline_list"])
                    loss += loss_func(outputs, labels["future"].to(config.device))
                    ade += torch.mean(get_ADE(outputs, labels["future"].to(config.device)))
                    t += 1
                if t > 0:
                    loss /= t
                    ade /= t
                    print("epoch:", epoch, "Mean metrics on eval dataset:", "loss:", loss, "ADE:", ade)
    torch.save(vector_net, os.path.join(config.model_save_path, save_name + '.model'))

if __name__ == '__main__':
    print("now device:", config.device)
    v_len = 9
    vector_net = VectorNetWithPredicting(v_len=v_len, time_stamp_number=30)
    random_dataloader = RandomDataloader(1, 0, 0, v_len)

#    train_model(vector_net, random_dataloader, torch.nn.MSELoss(), "random_model", is_print_test=True, epochs=200, decayed_factor=1)
    train_model(vector_net, random_dataloader, loss_func, "random_model", is_print_test=True, epochs=200, decayed_factor=1)

