import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np

ep=1e-64

class MIWAE(nn.Module):
    def __init__(self, h, d, K, p, classes):
        super(MIWAE, self).__init__()
        self.h = h # number of hidden units in (same for all MLPs)
        self.d = d # dimension of the latent space
        self.K = K # number of IS during training
        self.p = p # number of features
        self.classes = classes
        self.decoder = nn.Sequential(
            torch.nn.Linear(d, h),
            torch.nn.ReLU(),
            #torch.nn.Linear(h, h),
            #torch.nn.ReLU(),
            torch.nn.Linear(h, 3*p),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        )


        self.encoder = nn.Sequential(
            torch.nn.Linear(p, h),
            torch.nn.ReLU(),
            #torch.nn.Linear(h, h),
            #torch.nn.ReLU(),
            torch.nn.Linear(h, 2*d),  # the encoder will output both the mean and the diagonal covariance
        )
        
        self.classifier = nn.Sequential(
            torch.nn.Linear(p, h),
            torch.nn.ReLU(),
            #torch.nn.Linear(h, h),
            #torch.nn.ReLU(),
            torch.nn.Linear(h, self.classes)
        )
        
        self.p_z = td.Independent(td.Normal(loc=torch.zeros(d).cuda(),scale=torch.ones(d).cuda()),1)
        
    def miwae_loss(self, x, mask):
        #iota_x = x.clone().detach()
        iota_x = x
        #iota_x = x
        p_z = td.Independent(td.Normal(loc=torch.zeros(self.d).cuda(), scale=torch.ones(self.d).cuda()), 1)
        batch_size = iota_x.shape[0]
        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :self.d],\
            scale=torch.nn.Softplus()(out_encoder[..., self.d:(2*self.d)]) + ep),1)
  
        zgivenx = q_zgivenxobs.rsample([self.K])
        zgivenx_flat = zgivenx.reshape([self.K*batch_size,self.d])
  
        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.p]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.p:(2*self.p)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*self.p):(3*self.p)]) + 3
  
        data_flat = torch.Tensor.repeat(iota_x,[self.K,1]).reshape([-1,1])
        tiledmask = torch.Tensor.repeat(mask,[self.K,1])
  
        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape([-1,1]),\
            scale=all_scales_obs_model.reshape([-1,1]),\
            df=all_degfreedom_obs_model.reshape([-1,1])\
            ).log_prob(data_flat)
        
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K*batch_size,self.p])
  
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([self.K,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
  
        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq,0))
        #print(neg_bound)
  
        return neg_bound
        
    def miwae_impute(self, x, mask, L):
        iota_x = x
        #iota_x = x
        batch_size = iota_x.shape[0]
        p_z = td.Independent(td.Normal(loc=torch.zeros(self.d).cuda(), scale=torch.ones(self.d).cuda()), 1)
        out_encoder = self.encoder(iota_x)
        if torch.any(torch.isnan(out_encoder)):
            print('nan')
        if torch.any(torch.isnan(x)):
            print('NAN')
        q_zgivenxobs = td.Independent(td.Normal(loc=out_encoder[..., :self.d],\
            scale=torch.nn.Softplus()(out_encoder[..., self.d:(2*self.d)]) + ep),1)
  
        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L*batch_size,self.d])
  
        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :self.p]
        all_scales_obs_model = torch.nn.Softplus()(out_decoder[..., self.p:(2*self.p)]) + 0.001
        all_degfreedom_obs_model = torch.nn.Softplus()(out_decoder[..., (2*self.p):(3*self.p)]) + 3
  
        data_flat = torch.Tensor.repeat(iota_x,[L,1]).reshape([-1,1]).cuda()
        tiledmask = torch.Tensor.repeat(mask,[L,1]).cuda()
  
        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=all_means_obs_model.reshape([-1,1]),
        scale=all_scales_obs_model.reshape([-1,1]),df=all_degfreedom_obs_model.reshape([-1,1])).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L*batch_size,self.p])
  
        logpxobsgivenz = torch.sum(all_log_pxgivenz*tiledmask,1).reshape([L,batch_size])
        logpz = p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
  
        xgivenz = td.Independent(td.StudentT(loc=all_means_obs_model, scale=all_scales_obs_model, df=all_degfreedom_obs_model),1)

        imp_weights = torch.nn.functional.softmax(logpxobsgivenz + logpz - logq,0) # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L,batch_size,self.p])
        xm=torch.einsum('ki,kij->ij', imp_weights, xms) 
  

        return xm
        
    def miwae_mse(self, data, target):
        #imputed_data = self.miwae_impute(data, mask, L)
        imputed_data = data
        loss_fn = torch.nn.MSELoss(reduction='mean')
        mse = loss_fn(imputed_data, target)
        
        return mse
        
    def forward(self, data, mask):
        data_imputed = self.miwae_impute(data,mask,10)
        data_imputed[mask] = data[mask]
        logits = self.classifier(data_imputed)
        
        return data_imputed, logits
        
        
    

            
        
        
        
        
        
    
        
        