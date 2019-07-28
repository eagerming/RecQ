import os.path
class Config(object):
    def __init__(self, fileName):
        self.config = {}
        self.readConfiguration(fileName)

    def __getitem__(self, item):
        if not self.contains(item):
            raise Exception('parameter ' + item + ' is invalid!')

        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def update(self, another_config):
        self.config.update(another_config.config)

    def update_inferior(self, another_config):
        new_config = another_config.config.copy()
        new_config.update(self.config)
        self.config = new_config

    def __repr__(self):
        return self.config.__str__()

    def getOptions(self,item):
        if not self.contains(item):
            raise Exception('parameter '+item+' is invalid!')
        return self.config[item]

    def contains(self,key):
        return key in self.config

    def readConfiguration(self,fileName):
        # path = '../config/'+fileName
        path = fileName
        if not os.path.exists(path):
            print('config file {} is not found!'.format(path))
            raise IOError
        with open(path) as f:
            for ind,line in enumerate(f):
                if line.strip()!='':
                    try:
                        key,value=line.strip().split('=')
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d'%(ind))



class LineConfig(object):
    def __init__(self,content):
        self.content = content
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1

    def __repr__(self):
        return self.content

    def __getitem__(self, item):
        if not self.contains(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def getOption(self,key):
        if not self.contains(key):
            print('parameter '+key+' is invalid!')
            exit(-1)
        return self.options[key]

    def isMainOn(self):
        return self.mainOption

    def contains(self,key):
        return key in self.options


