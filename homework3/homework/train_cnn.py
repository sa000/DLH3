from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    num_epochs = 30
    start_time = time.time()
    loss_iteration = 0 
    train_data = load_data('data/train',128, transform=transformers)
    valid_data = load_data('data/valid')
    for epoch in range(num_epochs):
        training_accuracies, training_losses = [], []
        for image_batch, label_batch in iter(train_data):
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            x = classifier.forward(image_batch)
            p_y = torch.softmax(x, dim=1)        
                
            #Get Metric results on this batch
            cur_loss = l.forward(p_y, label_batch)
            cur_accuracy = accuracy(p_y, label_batch)
            training_accuracies.append(cur_accuracy.item())
            training_losses.append(cur_loss.item())
            #print("Current Loss:", loss_iteration, np.round(cur_loss.item(),5))
            #train_logger.add_scalar('loss', cur_loss.item(), global_step=loss_iteration) 
            #Reset for next iteration
            cur_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_iteration+=1
       # scheduler.step()

        training_accuracy = np.mean(training_accuracies)
        training_loss = np.mean(training_losses)
        #train_logger.add_scalar('accuracy', training_accuracy, global_step=loss_iteration) 
        
        time_so_far = np.round(time.time()-start_time,2)
        
        #Validation accuracy
        val_accuracies = []
        for img, label in valid_data:
            val_accuracies.extend(get_accuracy(classifier(img.to(device)), label).numpy())
        validation_accuracy = (sum(val_accuracies) / len(val_accuracies))
        #scheduler.step(validation_accuracy)
        #scheduler.step(np.mean(val_accuracies))
        #valid_logger.add_scalar('accuracy',validation_accuracy, global_step=loss_iteration) 
        print(epoch,training_accuracy.round(5), validation_accuracy.round(5),training_loss.round(5), time_so_far)  
        if validation_accuracy.round(5)>=.92:
            print('Saving Epoch', epoch)
            torch.save(classifier.state_dict(), 'cnn_%s.th'%epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
