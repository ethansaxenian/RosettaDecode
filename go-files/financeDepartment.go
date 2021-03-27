package factorymethod

type finance struct {
	department
}

func newFinance() iDepartment {
	return &finance{
		department: department{
			name:  "Finance Department",
			employees: 2,
		},
	}
}