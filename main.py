import warnings
warnings.filterwarnings(action="ignore")

import argparse
import click
import os
from fix_seed import seed_fix
from train import train
from pathlib import Path
from typing import Union

def main():
    # 명령줄 인자를 처리하기 위한 ArgumentParser를 생성합니다.
    parser = argparse.ArgumentParser()

    # 학습 데이터 경로를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--train_data', default='sample_train.zip', type=Path, help="path of directory containing training dataset")

    # 학습에 사용할 배치 크기를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # 학습 에포크 횟수를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')

    # 학습률을 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')

    # 학습 진행 상황을 보고할 간격을 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--report_interval', type=int, default=100, help='Report interval')

    # 생성자(Generator)에 입력될 노이즈 차원을 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--noise_dim', type=int, default=100, help= 'Input noise dimension to Generator')

    # 노이즈 투영 차원을 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--projection_dim', type=int, default=128, help= 'Noise projection dimension')

    # CLIP 임베딩 벡터 차원을 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--clip_embedding_dim', type=int, default=512, help= 'CLIP embedding vector dimension')

    # 학습 중 체크포인트를 저장할 경로를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--checkpoint_path', type=Path, default='model_exp1', help='Checkpoint path')

    # 생성된 이미지를 저장할 경로를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--result_path', type=Path, default='images_exp1', help='Generated image path')

    # 비조건적 손실을 사용할 것인지 여부를 결정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--use_uncond_loss', action="store_true")

    # 대조 손실을 사용할 것인지 여부를 결정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--use_contrastive_loss', action="store_true")

    # 생성자의 단계 수를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--num_stage', type=int, default=1)

    # 이전에 중단된 학습을 재개할 때 사용할 체크포인트 경로를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--resume_checkpoint_path', default=None)

    # 중단된 학습을 재개할 때 사용할 에포크 번호를 지정하는 명령줄 인자를 추가합니다.
    parser.add_argument('--resume_epoch', type=int, default=-1)

    # 명령줄 인자들을 파싱합니다.
    args = parser.parse_args()

    # 체크포인트와 생성된 이미지 저장 디렉토리를 생성합니다.
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    # 난수 시드를 고정하여 실험의 재현성을 보장합니다.
    seed_fix(0)

    # 학습 함수를 호출하여 모델을 학습시킵니다.
    train(args)

if __name__ == "__main__":
    main()
