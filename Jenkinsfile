node("jekins-agent1"){
    stage("Git clone"){
        sh 'cd /opt/jenkins/workspace/  && rm -rf TFpackegeText'
        sh 'git clone git@github.com:victorbai2/TFpackageText.git && tar -czvf package.tar.gz'
    }

    stage("code distribution"){
        sh 'scp package.tar.gz victor@192.168.1.5:/opt/'
        sh 'scp package.tar.gz victor@192.168.1.6:/opt/'
    }

    stage("model training"){
        sh 'ssh victorbai@192.168.1.5 "cd /opt/ && rm -rf TFpackageText && mkdir TFpackageText && tar -xvf package.tar.gz
        -C TFpackageText/ && rm -f package.tar.gz && cd TFpackageText/textMG && ./starup.sh -m train"'
    }

    stage("model distribution"){
        sh 'ssh victorbai@192.168.1.5 "scp -r ../save_model/multi_cnn_category_tf1/model_serving/* victor@192.168.1.6:/opt/model_serving"'
    }

    stage("start pred/infer service"){
        sh 'ssh victorbai@192.168.1.6 "cd /opt/ && rm -rf TFpackageText && mkdir TFpackageText && tar -xvf package.tar.gz
        -C TFpackageText/ && rm -f package.tar.gz && cd TFpackageText/textMG && ./api_run.sh restart
    }
}
