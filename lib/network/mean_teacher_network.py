import torch
import torch.nn as nn

class MeanTeacherNetwork(nn.Module):
    def __init__(self,
                 student_model,
                 teacher_model):
        super(MeanTeacherNetwork, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
            
        #  copy weight from student to teacher
        for t_param, s_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            t_param.data.copy_(s_param.data)  
            
        # turn off gradient for teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            param.detach_()
    
    def forward(self, inputs):
        # forward the teacher model
        with torch.no_grad():
            t_heatmap = self.teacher_model(inputs)
            
        # forward the student model
        s_heatmap  = self.student_model(inputs)
        
        return t_heatmap, s_heatmap
    
