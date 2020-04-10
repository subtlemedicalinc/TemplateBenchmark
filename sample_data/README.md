# FastMRI knee

Here are some sample data to download:

To download, we recommend using curl with recovery mode turned on:
```
curl -C - "https://fastmri-dataset.s3.amazonaws.com/singlecoil_train.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=u6Hifdar%2FQ5UCmMb9kUJJCxUi%2Bk%3D&Expires=1576183010" --output singlecoil_train.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/singlecoil_val.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=kYHhq9GYkBPHOIhgqhswj1qEitU%3D&Expires=1576183010" --output singlecoil_val.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/singlecoil_test_v2.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=kBGTqCDW82ZlHtNuqT8oGqG9vu8%3D&Expires=1576183010" --output singlecoil_test_v2.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/singlecoil_challenge.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=KqS5jyuQokFSfCbSOb2u42WqybY%3D&Expires=1576183010" --output singlecoil_challenge.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/multicoil_train.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=1sfiNdecQ%2F2Gn%2BnuwvbTYO6jh7M%3D&Expires=1576183010" --output multicoil_train.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/multicoil_val.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=JzQx4eW%2FrkvKw7USp3xQMdXcKzo%3D&Expires=1576183010" --output multicoil_val.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/multicoil_test_v2.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=OokhO1IlsiGRBFzRyxpYoyrEtx4%3D&Expires=1576183010" --output multicoil_test_v2.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/multicoil_challenge.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=hwwBrcIrXQFPHz7Wp9tINRPgrfU%3D&Expires=1576183010" --output multicoil_challenge.tar.gz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_mri_dicom_batch1.tar?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=7kg2uT3w%2FHKcsmGk1yi4fWxcEz4%3D&Expires=1576183010" --output knee_mri_dicom_batch1.tar
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_mri_dicom_batch2.tar?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=hRlsUdNB78I8qZEUWC0XEYQN3RE%3D&Expires=1576183010" --output knee_mri_dicom_batch2.tar
curl -C - "https://fastmri-dataset.s3.amazonaws.com/SHA256?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=BfO5bcJ5jvCUT80pqtDvkYrBOFI%3D&Expires=1576183010" --output SHA256
```
