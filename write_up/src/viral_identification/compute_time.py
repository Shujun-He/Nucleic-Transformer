
four_model=256**2/224**2*36+\
           288**2/224**2*20+20
four_model_time=2
eight_model=four_model+\
            30+\
            20*288**2/224**2+\
            32+\
            64*256**2/224**2
total_time=eight_model/four_model*four_model_time
print(total_time)
