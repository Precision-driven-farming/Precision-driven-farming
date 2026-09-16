# Requirements, Constraints and Risk Assessment

## 1. Problem Definition

Small-scale and commercial farmers have problems identifying crop diseases early and using pesticides in the right way. Crop diseases can spread quickly if they are not noticed early. This can damage crops, reduce the amount of food produced and cause farmers to lose money.

Normally, farmers inspect their fields manually, which can take a lot of time, especially on large farms. It can also be difficult to check every plant properly.

Farmers sometimes spray pesticides over the whole field even when only a small part of the field is affected. This can waste pesticides, increase farming costs and cause chemicals to enter the soil and nearby water sources.

Because of this, farmers need a faster and more effective way to monitor crops and identify diseases early.

## AI Benefit

AI can help by using cameras on drones or fixed poles to take pictures of crops and leaves. Computer vision can then be used to analyse these images and identify possible crop diseases.

The system can also create a spraying map that shows which areas may need treatment. This can help farmers use less pesticide, reduce costs and protect their crops.

For the municipality and local community, healthier crops can help increase local food production and support farmers. Using less pesticide can also help reduce chemical runoff into nearby water sources and support a cleaner environment.

## 2. Functional Requirements

The system should be able to:

1. Take pictures of crops using drones or fixed cameras.
2. Process the images using Python and OpenCV.
3. Find and analyse leaves in the images.
4. Identify crop diseases using a PyTorch AI model.
5. Identify the affected areas of the crop.
6. Give disease predictions soon after an image is taken.
7. Record the time and location where a disease is detected.
8. Use GPS information to connect disease detections to areas in the field.
9. Create a map showing areas where pesticide treatment may be needed.
10. Show the results to farmers through a simple interface.
11. Save previous results so farmers can compare crop health over time.
12. Give a confidence score or warning when a disease is detected.
13. Allow the farmer to check the results before making a spraying decision.

## 3. Non-Functional Requirements

| Requirement     | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| Accuracy        | The system should identify crop diseases correctly as much as possible.                                    |
| Performance     | The system should process images quickly enough for near-real-time monitoring.                             |
| Reliability     | The system should work properly during normal crop monitoring.                                             |
| Scalability     | The system should be able to work on both small and larger farms.                                          |
| Usability       | The system should be simple enough for farmers to understand and use.                                      |
| Security        | Farm information, images and user details should be protected from unauthorised access.                    |
| Maintainability | The software and AI model should be easy to update and fix when needed.                                    |
| Compatibility   | The system should work with the chosen cameras, drones, Python, OpenCV and PyTorch.                        |
| Availability    | The system should be available when crop monitoring is being carried out.                                  |
| Explainability  | The system should show the detected disease and, where possible, how confident the AI is about the result. |

## 4. Constraints

* **Limited training data:** It may be difficult to find enough good images of different crop diseases to train the AI.
* **Image quality:** Poor lighting, blurry pictures, shadows, wind and the position of leaves can affect the results.
* **Hardware limitations:** Drones and cameras have limits such as battery life, storage and processing power.
* **Internet connection:** Some farming areas may have weak or no internet connection.
* **Computing power:** Training the AI model may require a powerful computer or GPU.
* **Weather:** Rain, strong wind, fog and dust can make it difficult to collect good images.
* **Different crops:** Different crop types and growth stages may make disease detection more difficult.
* **Cost:** Drones, cameras, maintenance and computer equipment can be expensive.
* **Legal requirements:** Drone use and pesticide spraying must follow the relevant laws and regulations.
* **AI limitations:** The AI can make mistakes, especially when it sees images or diseases that were not included in its training data.

## 5. Risks and Mitigation

| Risk                             | Possible Impact                                                | How to Reduce the Risk                                                                            |
| -------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Incorrect disease classification | The farmer could use the wrong treatment.                      | Train the AI using a large range of good-quality images and allow the farmer to check the result. |
| False disease detection          | Pesticide may be used when it is not needed.                   | Use confidence levels and check uncertain results before spraying.                                |
| Missed disease detection         | A disease could spread to other crops.                         | Monitor crops regularly and improve the AI model over time.                                       |
| Poor image quality               | The AI may give incorrect results.                             | Use suitable cameras and check the quality of images.                                             |
| Drone failure                    | Some areas may not be monitored.                               | Maintain the drone and have another monitoring method available.                                  |
| Battery runs out                 | The drone may not finish checking the field.                   | Plan the flight properly and have charged spare batteries.                                        |
| Biased training data             | The AI may not work well on some crops or environments.        | Use different crops, diseases, environments and lighting conditions when training the model.      |
| Data security breach             | Farm information or images could be accessed by other people.  | Use passwords, access controls and secure data storage.                                           |
| Bad weather                      | Crop images may not be collected properly.                     | Monitor weather conditions and only fly when conditions are suitable.                             |
| Over-reliance on AI              | Farmers could make a bad decision based only on the AI result. | Use the AI as a support tool and allow human checking before important decisions.                 |
| Incorrect spraying map           | Pesticide could be sprayed in the wrong location.              | Check the GPS information and allow the farmer to verify the map.                                 |
| Regulatory problems              | The project may break drone or pesticide regulations.          | Make sure the system follows the relevant laws and regulations.                                   |

## Summary

This project uses AI to solve a real problem in the farming industry. The system uses computer vision and deep learning to help farmers find crop diseases and identify areas that may need treatment.

It does not replace the farmer but helps the farmer make better decisions.

By using pesticides more carefully, farmers can reduce waste and costs while protecting their crops. The project can also benefit the local community by supporting food production, reducing chemical pollution and helping the local farming economy.
