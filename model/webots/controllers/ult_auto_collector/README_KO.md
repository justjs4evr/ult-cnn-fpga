# Webots 자동 자료 수집

`worlds/ult_auto_collection.wbt`와 `worlds/ult_auto_collection_bright.wbt`는 e-puck을 지도상 지정 위치에 배치해 짧게 주행하는 수집 월드입니다. 후자는 방향광을 추가한 조명 환경입니다. 수동 수집 월드와 기존 원본 데이터는 그대로 사용합니다.

`worlds/ult_auto_track_01.wbt`와 `worlds/ult_auto_track_02.wbt`는 `track/autonomous_tracks_v2/`의 01·02번 지도를 실제 3m×3m 바닥으로 적용한 월드입니다. 각 지도에서 시험 촬영한 뒤 확인된 위치만 수집했습니다.

`worlds/ult_auto_case1_3m.wbt`와 `worlds/ult_auto_case1_3m_bright.wbt`는 첫 수집에 쓰인 case1 그림을 실제 3m×3m 바닥에 적용하고 좌·우 카메라를 바닥 쪽으로 조정한 월드입니다. 기존 1.5m 수집 월드와 에피소드를 구분해서 사용합니다. 검수·수집 결과는 `track/case1_3m/README_KO.md`에 기록했습니다.

## 실행 방법

1. 이 폴더에 `auto_collection_job.json`을 만듭니다. `mode`는 먼저 `preview`로 설정하고, 각 위치의 중앙 카메라 시작·종료 영상을 확인합니다.
2. 검수한 위치에만 `label`을 붙이고 `mode`를 `collect`로 바꿉니다. Webots에서 해당 월드를 열거나 시뮬레이션을 재설정하면 한 번 실행됩니다.
3. 결과는 `auto_collection_status.json`에서 확인합니다. 원래 명령은 `auto_collection_job_<episode_id>.json`으로 보존되며, 다음 재설정 때 자동으로 반복되지 않습니다.

명령 예시:

```json
{
  "mode": "collect",
  "world": "ult_auto_collection.wbt",
  "environment": "baseline",
  "arena_size_m": 1.5,
  "poses": [
    {
      "id": "verified_straight",
      "map_x_px": 350,
      "map_y_px": 1850,
      "heading_deg": 90,
      "duration_s": 1.5,
      "left_speed": 1.2,
      "right_speed": 1.2,
      "label": "straight"
    }
  ]
}
```

지도 좌표는 3000×3000 픽셀 기준입니다. `arena_size_m`은 바닥의 실제 한 변 길이이며 기존 수집 월드는 1.5, `autonomous_tracks_v2` 월드는 3.0으로 지정합니다. 방향은 지도에서 오른쪽 0°, 위쪽 90°, 왼쪽 180°입니다. 수집기는 바퀴 하나 이상이 1.5cm 이상 굴러가고 최소 0.4초가 지난 뒤 촬영하므로 제자리 회전도 기록할 수 있습니다. 중앙 카메라 아래쪽에 밝은 차선 픽셀이 거의 없으면 건너뜁니다. 자동으로 클래스의 의미를 판단하지는 않으므로 `preview` 검수가 필요합니다.

원본 3카메라 영상은 `../ult_data_collector/dataset/images/`에, 라벨은 `labels.csv`에 누적됩니다. `auto_pose_manifest.csv`에는 지도 및 월드 위치, 방향, 밝은 차선 픽셀 수가 남습니다. 실행별 조명과 라벨 출처는 `episode_<episode_id>.json`에 기록됩니다. 새 자료를 모은 뒤 `prepare_ult_dataset.py`로 학습 목록을 갱신하세요. 같은 위치를 반복 촬영한 장면은 매우 비슷하므로 검증 데이터는 프레임 단위가 아니라 에피소드 또는 환경 단위로 분리해야 합니다.

차선 복귀처럼 연속 동작이 필요한 경우에는 `mode: "trajectory_preview"`를 사용합니다. 각 pose에 시작 좌표와 `segments` 목록을 주면 세그먼트마다 바퀴 속도와 시간을 적용하면서 일정 간격(`capture_interval_s`, 기본 0.25초)으로 3카메라 영상을 저장합니다. 결과 영상은 `preview/<episode_id>/`, 실제 이동 좌표와 촬영 시점은 `auto_collection_status_<episode_id>.json`에 남습니다. 이 모드는 `labels.csv`나 학습 목록에 프레임을 추가하지 않습니다. 회전 강도 라벨은 e-puck 바퀴 속도로 자동 결정하지 않고 실차 조향 기준과 연속 영상을 검수해 정해야 합니다.

공식 레인 타일 기반 연습 월드 `ult_auto_iro_lane_01.wbt`~`14.wbt`도 같은 수집기를 사용합니다. `arena_size_m`은 3.0입니다. 이 월드의 좌·우 보조 카메라는 바닥이 보이도록 아래로 기울어져 있습니다. 11·12번은 직각 코너에서 급좌·급우회전 자료를 촬영하는 월드이고, 13·14번은 이를 연속 코스에 넣은 월드입니다. 수집기는 좌·우 바퀴의 움직임을 각각 확인하므로 제자리 회전 중에도 장면이 바뀌면 저장할 수 있습니다. 카메라 교정 전 시험 에피소드 등 학습에서 뺄 자료는 원본을 지우지 않고 데이터셋의 `excluded_episodes.json`에 에피소드 ID를 적습니다. `prepare_ult_dataset.py`가 이 목록을 제외하고 학습 목록을 만듭니다.
