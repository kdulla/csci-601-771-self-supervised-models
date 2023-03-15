import numpy as np
import matplotlib.pyplot as plt

BERT_base_uncased_dev = [0.7058103975535168, 0.7149847094801223, 0.7128440366972477,
                         0.6217125382262997, 0.6217125382262997, 0.6217125382262997,
                         0.6217125382262997, 0.6217125382262997, 0.6217125382262997]
BERT_base_uncased_test = [0.7098808689558515, 0.677645409950946, 0.7000700770847933,
                          0.6348983882270498, 0.6040644709180099, 0.6222845129642607,
                          0.6173791170287316, 0.6250875963559915, 0.6229852838121934]

BERT_base_cased_dev = [0.6737003058103975, 0.6217125382262997, 0.6929663608562691,  
                       0.6217125382262997, 0.6217125382262997, 0.6217125382262997,
                       0.6217125382262997, 0.6217125382262997, 0.6217125382262997]
BERT_base_cased_test = [0.6748423265592152, 0.6320953048353188, 0.6832515767344078,  
                        0.6215837421163279, 0.6398037841625789, 0.629292221443588,
                        0.6370007007708479, 0.6166783461807989, 0.6229852838121934]

barWidth = 0.2
br1 = np.arange(len(BERT_base_uncased_test))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

plt.bar(br1, BERT_base_uncased_dev, color ='tab:orange', width = barWidth,
        edgecolor ='grey', label ='BERT-base-uncased dev accuracy')
plt.bar(br2, BERT_base_uncased_test, color ='tab:blue', width = barWidth,
        edgecolor ='grey', label ='BERT-base-uncased test accuracy')

plt.bar(br3, BERT_base_cased_dev, color ='tab:red', width = barWidth,
        edgecolor ='grey', label ='BERT-base-cased dev accuracy')
plt.bar(br4, BERT_base_cased_test, color ='tab:cyan', width = barWidth,
        edgecolor ='grey', label ='BERT-base-cased test accuracy')

plt.xlabel('Hyperparameters', fontweight ='bold', fontsize = 15)
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15, rotation=0)
plt.xticks([r + barWidth for r in range(len(BERT_base_uncased_test))],
        ['lr=1e-4\n epochs=5','lr=1e-4\n epochs=7','lr=1e-4\nepochs=9',
         'lr=5e-4\n epochs=5','lr=5e-4\n epochs=7','lr=5e-4\nepochs=9',
         'lr=1e-3\n epochs=5','lr=1e-3\n epochs=7','lr=1e-3\nepochs=9'], rotation=90)
plt.ylim(0.5, 0.75)
plt.legend()
plt.show()