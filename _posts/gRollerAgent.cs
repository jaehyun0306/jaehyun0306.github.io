using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class gRollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public GameObject Check1;
    public GameObject Check2;
    public GameObject Check3;
    public GameObject Check4;
    public GameObject Check5;
    public GameObject Check6;
    public GameObject Check7;
    public GameObject Check8;

    public GameObject viewModel = null;

    public override void OnEpisodeBegin()
    {
        //새로운 에피소드 시작시, 다시 에이전트의 포지션의 초기화
        // if the Agent fell, zero its momentum

            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(-24.5f, 0.5f, -24.5f);

        Target.localPosition = new Vector3(25.5f, 0.5f, 5.5f);
        Check1.SetActive(true);
        Check2.SetActive(true);
        Check3.SetActive(true);
        Check4.SetActive(true);
        Check5.SetActive(true);
        Check6.SetActive(true);
        Check7.SetActive(true);
        Check8.SetActive(true);

    }

    /// <summary>
    /// 강화학습을 위한, 강화학습을 통한 행동이 결정되는 곳
    /// </summary>
    public float forceMultiplier = 10;
    float m_LateralSpeed = 1.0f;
    float m_ForwardSpeed = 1.0f;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);
        AddReward(-1 / MaxStep);
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var forwardAxis = act[0];
        var rotateAxis = act[1];

        switch (forwardAxis)
        {
            case 1:
                dirToGo = transform.forward * m_ForwardSpeed;
                break;
        }

        switch (rotateAxis)
        {
            case 1:
                rotateDir = transform.up * -1f;
                break;
            case 2:
                rotateDir = transform.up * 1f;
                break;
        }

        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        rBody.AddForce(dirToGo * forceMultiplier, ForceMode.VelocityChange);
    }

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Target"))
        {
            SetReward(+1.0f);
            EndEpisode();
        }

        if (collision.gameObject.CompareTag("Wall"))
        {
            SetReward(-1.0f);
            EndEpisode();
        }

        if (collision.gameObject.CompareTag("Check"))
        {
            AddReward(0.1f);
            collision.gameObject.SetActive(false);
        }
    }

    /// <summary>
    /// 해당 함수는 직접조작 혹은 규칙성 있는 코딩으로 조작시키기 위한 함수
    /// </summary>
    /// <param name="actionOut"></param>

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut.Clear();
        //forward
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }

        //rotate
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 2;
        }
    }
}
